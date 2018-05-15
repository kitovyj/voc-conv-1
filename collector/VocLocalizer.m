function [StimPosEst,MidDist,SCorrFinal,CorrTime,DeltaTime,CorrMidDist] = VocLocalizer(varargin)
% Localize vocalizations in space using a microphone array
%
% Realignment: 29.9.2016 (Jesse/Bernhard)
% - We think the intended distance between the lower edges of the microphones was
%   - Horizontal: 46cm, i.e. 23cm from the center (this corresponds to the inner edges of the rear platform walls) 
%   - Vertical :     15cm above the upper end of the platform 
%                      = 34.4cm above the platform level
%   - Depth : centered over the platforms
%   - Microphone angle : 45 degrees
%   NOTE: This means that the membranes of the microphones are at a
%   different location. Since the diameter of the microphone front is 36mm,
%   the center of the membrane is located 1.27cm further up and inward.
%  This means the position of the MEMBRANES is 
%  - Horizontal: 43.46cm, i.e. 21.73cm from the center (this corresponds to the inner edges of the rear platform walls) 
%   - Vertical :     16.27cm above the upper end of the platform 
%                      = 35.67cm above the platform level
%   - Depth : centered over the platforms
%   - Membrane angle : 45 degrees
%  The Membrane position should be the authorative position of the
%  microphones.
% 
% The Height of the speaker in relation to the microphones was remeasured as 31.4cm
%
% Old Notes on Position:
% MicDist = 0.46; %m % Measurements from new setup
% - Estimate of previous distances : 0.38
% MicHeight = 0.314; %m % Measurement of Speaker
% MicHeight = 0.334; %m % Measurement for Mouse
% - Estimate of previous height: 0.35
% Platform is assumed to be at Z = 0.
%
% Lenses Used:
% - 12.5mm : leads to XXX mm
% - 35 mm (XR300, SainSonic) : leads to 56mm 


%% PARSE ARGUMENTS
P = parsePairs(varargin);
checkField(P,'Sounds')
checkField(P,'SR')
checkField(P,'NFFT',250);
checkField(P,'Offset',1);
checkField(P,'HighPass',25000);
checkField(P,'CorrTime',0.0001); % 0.1 ms
checkField(P,'CenterShift',0.009); %10mm of CenterShift based on measured discrepancy between position of left and right microphone
checkField(P,'GainCorrection',1.2);
checkField(P,'LocMethod','Geometric');
checkField(P,'RemoveAmplitude',0);
checkField(P,'CorrMethod','GCC');
checkField(P,'MaxMethod','Max');
checkField(P,'IntersectionMethod','Lines');
checkField(P,'DelayRange',0.0001); %75us
checkField(P,'MicrophonePositions',{[-0.23,0,0.314],[0.23,0,0.314]})
checkField(P,'SourceHeight',0);
checkField(P,'Estimation',[]); % X , Y , or XY
checkField(P,'StimulusPosition',[]); % For checking in simulated recordings
checkField(P,'FreqSelect',1.5);
checkField(P,'FIG',1); %
checkField(P);

%% PREPARE ANALYSIS
NMic = length(P.MicrophonePositions);
for iM=1:NMic; 
  P.Sounds{iM} = vertical(P.Sounds{iM}); 
  P.MicrophonePositions{iM}(3) = P.MicrophonePositions{iM}(3) - P.SourceHeight; 
end
NSteps = length(P.Sounds{1});
if ischar(P.CorrMethod) P.CorrMethod = {P.CorrMethod}; end
CorrSteps = round(P.CorrTime*P.SR);

%% POTENTIALLY FILTER THE DATA
if P.HighPass
  [b,a]=butter(4,P.HighPass/(P.SR/2),'high');
  for i=1:length(P.Sounds);  P.Sounds{i} = filter(b,a,P.Sounds{i}); end
end

if P.RemoveAmplitude
  for i=1:length(P.Sounds)
    H{i} = abs(hilbert(P.Sounds{i})); P.Sounds{i} = P.Sounds{i}./H{i};
  end
end

%% COMPUTE CENTER DISTANCE FOR EACH PAIR OF MICROPHONES
MicPairs = {};
for iM1 = 1:NMic
  for iM2 = iM1+1:NMic

    SCorrFinal = ones(2*CorrSteps+1,1);
    for iM = 1:length(P.CorrMethod)
      switch P.CorrMethod{iM}
        case 'Specgram';
          % COMPUTE SPECTROGRAM OF THE SOUND FROM TWO MICROPHONES
          S1 = HF_specgram(single(P.Sounds{iM1}),P.NFFT,P.SR,[],P.NFFT-P.Offset,0,0);
          S2 = HF_specgram(single(P.Sounds{iM2}),P.NFFT,P.SR,[],P.NFFT-P.Offset,0,0);
          
          % SELECT FREQUENCY INDICES TO COMPUTE THE CROSSCORRELATION
          FreqMarginals = mean(abs(S1)+abs(S2),2);
          FreqInds = find( FreqMarginals > P.FreqSelect*median(FreqMarginals));
          
          % COMPUTE CROSSCORRELATIONS
          Corrs= zeros(length(FreqInds),2*CorrSteps+1);
          for iF=1:length(FreqInds)
            cInd = FreqInds(iF);
            Corrs(iF,:) = xcorr(abs(S1(cInd,:)),abs(S2(cInd,:)),CorrSteps,'unbiased');
          end
          Corrs = Corrs./repmat(FreqMarginals(FreqInds),1,size(Corrs,2));
          
          cSCorr  = mean(Corrs)';
          DT = [-CorrSteps:CorrSteps]/P.SR;
          CorrKernel = 0.01*exp(-DT.^2/(2*0.00065^2)) + 1;
          cSCorr = cSCorr / max(cSCorr);
          cSCorr = cSCorr.*CorrKernel';          
          SCorr{iM} =cSCorr;
          
        case 'GCC';
          NFFT = NSteps;
          S1F = fft(double(P.Sounds{iM1}),NFFT); S2F = fft(double(P.Sounds{iM2}),NFFT);
          F = [1:NFFT]/NFFT*P.SR; F = F(1:round(NFFT/2));
          Rotator = S1F.*conj(S2F)./(abs(S1F).*abs(conj(S2F)));
          XCorr = ifft(Rotator);
          SCorr{iM} = abs( hilbert([ XCorr(round(NFFT/2)+1:end) ; XCorr(1:round(NFFT/2))  ] ) );
          SCorr{iM} = SCorr{iM}(round(end/2)-CorrSteps:round(end/2)+CorrSteps);
                    
        case 'XCorr';
          SCorr{iM} = xcorr(P.Sounds{iM1},P.Sounds{iM2},CorrSteps,'unbiased');
          SCorr{iM} = abs(hilbert(SCorr{iM}));
          
        case 'Micro';
          OversampleFactor = 8;
          % OVERSAMPLE DATA
          SoundsLong{1} = interpft(P.Sounds{iM1},NSteps*OversampleFactor+1);
          SoundsLong{2} = interpft(P.Sounds{iM2},NSteps*OversampleFactor+1);
          P.SR = P.SR*OversampleFactor;
          
          % REMOVE PHASE
          for i=1:length(SoundsLong)
            H{i} = abs(hilbert(SoundsLong{i}));
            SoundsLong{i} = SoundsLong{i}./H{i};
          end
          
          % CONSTRUCT HISTOGRAM OF THE PHASE MATCHES
          CorrSteps = round(P.CorrTime*P.SR)/2;
          WindowSteps = P.CorrTime*P.SR;
          NFFT = WindowSteps;
          MaxHist = zeros(2*CorrSteps+1,1);
          Steps = round(linspace(1,NSteps-WindowSteps,100));
          for i=1:length(Steps)
            cStep = Steps(i);
            cInd = cStep+[0:WindowSteps-1];
            S1 = SoundsLong{1}(cInd); S2 = SoundsLong{2}(cInd);
            X = xcorr(S1,S2,CorrSteps,'biased');
            X(X<max(X)/2) = -inf;
            [StimPosEst,Vals] = findLocalExtrema(X,'max');
            MaxHist(StimPosEst) = MaxHist(StimPosEst) + 1;
          end
          SCorr{iM} = MaxHist;
          
        otherwise error(['Method ',P.CorrMethod,' not known.']);
      end
      
      % COMBINE THE ESTIMATES FROM DIFFERENT METHODS
      SCorrFinal = SCorr{iM}.*SCorrFinal;
    end
    
    %% POST PROCESS LOCALIZATION
    % SMOOTH THE CORRELATION
    SCorrFinal = relaxc(SCorrFinal,1);
    
    % SUBSELECT A RANGE FOR CHOOSING THE RIGHT DELAY
    CorrTime = [-CorrSteps:CorrSteps]/P.SR;
    NotInd = find(abs(CorrTime)>P.DelayRange);
    SCorrRange= SCorrFinal;
    SCorrRange(NotInd) = -inf;
    
    % EXTRACT CENTER OF CORRELATION
    switch P.MaxMethod
      case 'Weighted';
        StimPosEst = sum([1:length(SCorrRange)]'.*SCorrRange./(sum(SCorrRange)));
      case 'Max';
        [M,StimPosEst] = max(SCorrRange);
      otherwise error('Maximum Method not known');
    end
    
    DeltaTime(iM1,iM2) = real((StimPosEst-(CorrSteps+1))/P.SR); % DeltaTime is negative, if signal arrives first on the left (channel 1), i.e. position is also on the left
    
    MidDist(iM1,iM2) = LF_translateTime2Space(DeltaTime(iM1,iM2),P.LocMethod,P.CenterShift,P.GainCorrection,P.MicrophonePositions);
    
    if abs(imag(MidDist(iM1,iM2)))>0
      MidDist(iM1,iM2) = NaN;
      fprintf('Warning: Out of bounds distance!\n');
    end
    
    CorrMidDist{iM1,iM2} = LF_translateTime2Space(CorrTime,P.LocMethod,P.CenterShift,P.GainCorrection,P.MicrophonePositions([iM1,iM2]));
    
    MicPairs{end+1} = [iM1,iM2];
    SCorrAll{iM1,iM2} = SCorrFinal;
  end
end

%% FUNCTION TO TRANSLATE ON THE CENTER TO OFFCENTER
% correction factor 1.05 apparently necessary, but not explained.
InvFun = @(MidDist,OffCenter,MicSep)MidDist * sqrt((4*(OffCenter).^2 + MicSep.^2 - MidDist.^2)./(MicSep.^2 - MidDist.^2));
MicPos = cell2mat(P.MicrophonePositions');
MIN = min(MicPos) - 0.05; MAX = max(MicPos) + 0.05;
Range = sqrt(2)*max(MAX-MIN);
OrthDist = [-Range:0.001:Range];
MicHeight = unique(MicPos(:,3)); 
if length(MicHeight) > 1; error('Microphones are assumed to be at the same height'); end
PlatformDist = sqrt(OrthDist.^2 + MicHeight^2); % Absolute, orthogonal distance to the shifted line of potential positions
Intersections = [];

if isempty(P.Estimation)
  if length(unique(MicPos(:,1))==1); P.Estimation = 'Y'; end;
  if length(unique(MicPos(:,2))==1); P.Estimation = 'X'; end;
  if length(unique(MicPos(:,1)))>1 && length(unique(MicPos(:,2)))>1; P.Estimation = 'XY'; end
end

%% COMPUTE THE SPATIAL LOCATION FROM THE VARIOUS MEASUREMENTS
% Compute the lines defined by each Microphone Pair
for iP = 1:length(MicPairs)
  P1 = P.MicrophonePositions{MicPairs{iP}(1)}(1:2);
  P2 = P.MicrophonePositions{MicPairs{iP}(2)}(1:2);
  % Find Orthogonal Vector onto Line between Microphones
  VA = (P2-P1); VA = VA/norm(VA);
  VAO = [VA(2),-VA(1)]; VAO = VAO/norm(VAO);
  % Set Starting Point based on MidDist
  cMidDist = MidDist(MicPairs{iP}(1),MicPairs{iP}(2));
  Lines(iP).P0 = (P1 + P2)/2;
  Lines(iP).P1 = P1;
  Lines(iP).P2 = P2;
  Lines(iP).V = VAO;
  Lines(iP).Mics = MicPairs{iP};
  Lines(iP).MidDist = cMidDist;
  Lines(iP).MicSep = norm(P2-P1);
  
  % ROTATION MATRIX for each pair of microphones 
  % using the orthogonal vector onto the connecting line
  alpha = angle(Lines(iP).V(1) + sqrt(-1)*Lines(iP).V(2));
  Rotator = [cos(alpha),-sin(alpha);sin(alpha),cos(alpha)];
  
  Lines(iP).MidDists = InvFun(Lines(iP).MidDist,PlatformDist,Lines(iP).MicSep);
  Lines(iP).MidDistsRot = Rotator*[OrthDist;Lines(iP).MidDists] + repmat([Lines(iP).P0(1);Lines(iP).P0(2)],[1,length(OrthDist)]);
end
R.Lines = Lines; R.MIN = MIN; R.MAX = MAX;

%% ESTIMATE OVERLAP OF ESTIMATION LINES
switch P.IntersectionMethod
  case 'Lines';
    switch P.Estimation
      case 'XY'; NStepsX = 2001; NStepsY=2001;
      case 'X'; NStepsX = 5001; NStepsY = 11; % MIN(2)=-1e-3; MAX(2) = 1e-3; 
      case 'Y'; NStepsY = 5001; NStepsX = 11; % MIN(1)=-1e-3; MAX(1) = 1e-3;
    end        
    mX = linspace(MIN(1),MAX(1),NStepsX);
    mY = linspace(MIN(2),MAX(2),NStepsY);
    M = zeros(length(mY),length(mX));
    % FILL A MATRIX
    for iP=1:length(MicPairs)
      cP = Lines(iP).MidDistsRot;
      i1 = ceil((cP(2,:)-MIN(2))/(MAX(2)-MIN(2))*NStepsY);
      i2 = ceil((cP(1,:)-MIN(1))/(MAX(1)-MIN(1))*NStepsX);
      SelInd = i1>0 & i1<=NStepsY & i2>0 & i2<=NStepsX;
      i1 = i1(SelInd); i2 = i2(SelInd);
      Ind = sub2ind(size(M),i1,i2);
      M(Ind)  = M(Ind) + 1;
    end
    M = relaxc2(M,10);
    R.M = M; R.mX = mX; R.mY = mY;
    
  case 'Intersections';
    % Compute the intersection points between all pairs of lines
    % Complicated to numerically find the intersections
    
end

%% EXTRACT PROJECTIONS ONTO THE ESTIMATED DIRECTIONS
switch P.Estimation
  case 'X';
    iY = find(mY==min(abs(mY)));
    iX = find(M(iY,:)==max(M(iY,:)));
    StimPosEst(1) = mean(mX(iX));
    StimPosEst(2) = 0;
  case 'Y';
    iX = find(mX==min(abs(mX)));
    iY = find(M(iX,:)==max(M(iX,:)));
    StimPosEst(2) = mY(iY);
    StimPosEst(1) = 0;
  case 'XY';
    [iY,iX] = find(M==max(M(:)));
    StimPosEst(1) = mean(mX(iX));
    StimPosEst(2) = mean(mY(iY));
end

if ~isempty(P.StimulusPosition)
  Error = norm(StimPosEst - P.StimulusPosition(1:2));
  fprintf(['Error = ',num2str(Error*1000),'mm (Estimated: ',num2str(StimPosEst),'mm)\n']);
end

if ~isempty(P.FIG) && P.FIG ~= 0 
  R.Sounds = P.Sounds;
  R.Time = [1:NSteps]/P.SR;
  R.SCorr = SCorrAll;
  R.CorrTime = CorrTime;
  LF_showLocalization(P.StimulusPosition,StimPosEst,P.MicrophonePositions,R,P); 
end
end % END OF MAIN FUNCTION

%% COMPUTE POSITION FROM TIMING
function MidDist = LF_translateTime2Space(DeltaTime,LocMethod,CenterShift,GainCorrection,MicrophonePositions)

VSound = 343; %m/s
DeltaDist = VSound * DeltaTime;
switch LocMethod
  case 'Geometric';
    % COMPUTE POSITION BASED ON SETUP GEOMETRY AND CORRELATION
    if MicrophonePositions{1}(3) ~= MicrophonePositions{2}(3); error('Microphones need to be at the same height'); end 
    MicHeight = MicrophonePositions{1}(3);
    MicDist = MicrophonePositions{2}(1)-MicrophonePositions{1}(1);
    T = DeltaDist;   H = MicHeight;  D  = MicDist;
    
    MidDist = 0.5.*T.* sqrt((4*H^2 + D^2 - T.^2)./(D^2-T.^2));   
    MidDist = MidDist - CenterShift;
     
  case 'Empirical';
    MidDist = (DeltaDist - CenterShift)/GainCorrection;
    
  case 'Linear';
    MidDist = DeltaDist/2;

  otherwise error(['Localization Method ',LocMethod,'not known.']);
end
end

%% SHOW THE LOCALIZATION RESULTS
function Pos = LF_showLocalization(StimPos,StimPosEst,MicPos,R,P)

figure(P.FIG); clf; [DC,AH] = axesDivide(1,1,'c');
axes(AH(1)); hold on;

% PLOT MICROPHONES & STIMULUS
NMic = length(MicPos);
for iM = 1:NMic
  plot3(MicPos{iM}(1),MicPos{iM}(2),MicPos{iM}(3),'.r','MarkerSize',20,'HIttest','off');
  text(MicPos{iM}(1),MicPos{iM}(2),MicPos{iM}(3),['  M',num2str(iM)],...
    'ButtonDownFcn',{@LF_showData,R.Time,R.Sounds{iM}});
  plot3([MicPos{iM}(1),MicPos{iM}(1)],[MicPos{iM}(2),MicPos{iM}(2)],[0,MicPos{iM}(3)],'--','Color',[0.5,0.5,0.5],'HIttest','off');
end
if ~isempty(StimPos)
  plot3(StimPos(1),StimPos(2),StimPos(3),'.g','MarkerSize',20,'HIttest','off');
end
axis square; hold on; box on; grid on;
axis([R.MIN(1),R.MAX(1),R.MIN(2),R.MAX(2),0,R.MAX(3)]) ;

% PLOT THE INTERSECTION LINES
for iP = 1:length(R.Lines)
  cLine = R.Lines(iP);
  plot3([cLine.P1(1),cLine.P2(1)],[cLine.P1(2),cLine.P2(2)],[0,0],'--','Color',[0.5,0.5,0.5],'HIttest','off');
  plot3([cLine.P1(1),cLine.P2(1)],[cLine.P1(2),cLine.P2(2)],...
    [MicPos{cLine.Mics(1)}(3),MicPos{cLine.Mics(2)}(3)],'--','Color',[0.5,0.5,0.5],'HIttest','off');
  cP = cLine.MidDistsRot;
  plot(cP(1,:),cP(2,:),'Color',[0.5,0.5,0.5],'HIttest','off');
  text(cP(1,round(end/2)),cP(2,round(end/2)),['M',num2str(cLine.Mics(1)),' - M',num2str(cLine.Mics(2))],...
    'ButtonDownFcn',{@LF_showData,R.CorrTime,R.SCorr{cLine.Mics(1),cLine.Mics(2)}});
end

% PLOT THE POSITION ESTIMATION BASIS
switch P.IntersectionMethod
  case 'Lines';
    imagesc(R.mX,R.mY,R.M,'Hittest','off');
  case 'Intersections';
end

% PLOT ESTIMATED STIMULUS
plot3(StimPosEst(1),StimPosEst(2),0.000001,'.k','MarkerSize',20);

colormap(HF_colormap({[1,1,1],[1,0,0]})); colorbar
switch P.Estimation;
  case 'XY'; view(20,20);
  otherwise view(0,90);
end
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
end

%% CALLBACK TO PLOT THE SOUNDS / CROSSCORRELATIONS
function LF_showData(H,E,X,Y)

SelType = get(gcf,'SelectionType');
switch SelType
  case 'normal'; FIG = 1000; figure(FIG);
  case {'alt','extend'}; FIG = round(1e6*rand); figure(FIG)
end

% SHOW RASTER PLOT
plot(X,Y);
end