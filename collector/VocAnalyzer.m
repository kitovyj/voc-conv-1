function Vocs = VocAnalyzer(Vocs,varargin)
% COLLECT PROPERTIES AND PERFORM INITIAL ANALYSIS OF VOCALIZATIONS  

P = parsePairs(varargin);
checkField(P,'CorrDur',0.001);
checkField(P,'Method','PCA'); % 
checkField(P,'MaxDims',3); % 
checkField(P,'Selector',{});
checkField(P,'BaseData',{});
checkField(P,'WithinPhrasePause',0.3);
checkField(P,'ComputeLocation',1);
checkField(P,'MicrophonePositions',[]);
checkField(P,'SourceHeight',0);
checkField(P,'CenterShift',0)
checkField(P,'CheckProperties',0);
checkField(P,'Selection',[1:length(Vocs)])
checkField(P,'FIG',0); % 

Vocs = vertical(Vocs);
IVIs = [Vocs.Interval];
NMics = length(Vocs(1).Sound);
if NMics == 1; P.ComputeLocation = 0; end

% FIND PHRASES
PhraseNumberByVoc = zeros(size(Vocs));
PhraseLengthByVoc = zeros(size(Vocs));
PhraseCounter = 1; PhraseLength = 0;
for iV=1:length(IVIs)
  if IVIs(iV) < P.WithinPhrasePause 
    PhraseLength = PhraseLength+1;
    PhraseNumberByVoc(iV) = PhraseCounter;
  else
    PhraseCounter = PhraseCounter + 1;
    PhraseNumberByVoc(iV) = PhraseCounter;
    PhraseLengthByVoc(iV-PhraseLength:iV-1) = PhraseLength;
    PhraseLength = 1;
  end
end

% COMPUTE SIMPLE PROPERTIES OF THE VOCALIZATIONS
for iV=P.Selection
  fprintf([num2str(iV),' ']); 
  SRSpec = Vocs(iV).SRSpec;
  SRSound = Vocs(iV).SRSound;
  
  % COMPUTE AVERAGE SPECTRAL ENERGY
  for iS =1:NMics  
    Vocs(iV).MeanEnergy(iS) =  mean(mat2vec(Vocs(iV).Spec{iS})); 
  end
  % FIND LOUDER SIDE
  [~,iLarger] = max([Vocs(iV).MeanEnergy]);
 
   % REMOVE OUTLIERS
  cSpec = Vocs(iV).Spec{iLarger};
  FreqHist = sum(cSpec,2);
  CFH = cumsum(FreqHist);
  cPosMedian = find(CFH>=CFH(end)*0.5,1,'first');
  cLowRange = cPosMedian - find(CFH>=CFH(end)*0.25,1,'first');
  cHighRange = find(CFH>=CFH(end)*0.75,1,'first') - cPosMedian;
  NullInds = [1:cPosMedian - 5*cLowRange-2, cPosMedian + 3*cHighRange+2:size(cSpec,1)];
  cSpec(NullInds,:)= 0;
  % REMOVE SPECLE NOISE
  cSpec = medfilt2(cSpec,[3,3]);
  
  % COMPUTE TIME STEPS OF ACTUAL VOCALIZATION
  % SPECTRUM
  cProfile = max(cSpec,[],1);
  iSpecStart = find(cProfile>max(cProfile)*0.15,1,'first');
  iSpecStop = find(cProfile>max(cProfile)*0.15,1,'last');
  VocRange = iSpecStart : iSpecStop;
  SpecSteps = size(Vocs(iV).Spec{1},2);
  % REASSIGN TIME RANGE
  StartTime = iSpecStart/SRSpec;
  PreTime = StartTime;
  StopTime = iSpecStop/SRSpec;
  PostTime = StopTime;
  Vocs(iV).StartTime = StartTime;
  Vocs(iV).StopTime = StopTime;
  Vocs(iV).Duration = StopTime-StartTime;
  
 % SOUND WAVEFOR
  SoundSteps = length(Vocs(iV).Sound{1});
  iSoundStart = round(StartTime*SRSound)+1;
  iSoundStop = min(round(StopTime*SRSound), length(Vocs(iV).Sound{1}));  % ON SOME VOCS (Mouse 3 Rec 80 Voc 13) THE STOPTIME IS LARGER THAN VOCALIZATION LENGTH. R?mi 02/05/16
  
  % COMPUTE AVERAGE LEVEL BASED ON SOUND
  for iM=1:NMics
    Vocs(iV).Level(iM) = std(Vocs(iV).Sound{iM}(iSoundStart:iSoundStop));
    Vocs(iV).BaselineStd(iM) = std(Vocs(iV).Sound{iM}([1:iSoundStart-1,iSoundStop:end]));
    Vocs(iV).SNR(iM) = Vocs(iV).Level(iM)/Vocs(iV).BaselineStd(iM);
  end
  
  Vocs(iV).BaselineStd(iM) = std(Vocs(iV).Sound{iM}(1:iSoundStart));
  Vocs(iV).SNR(iM) = Vocs(iV).Level(iM)/Vocs(iV).BaselineStd(iM); 
  
  % TAKE OUT SPECTRUM AGAIN
  cSpec = cSpec(:,VocRange);
  
  % COMPUTE THE SPECTRAL LINE (= maximal frequency in every bin)
  cSpecLine = NaN*zeros(1,length(VocRange));
  for iP = 1:size(cSpec,2) % STEP ALONG TIME
    if any(cSpec(:,iP))
      CS  = cumsum(cSpec(:,iP)) ; cPos = find(CS>=CS(end)/2,1,'first');
       cSpecLine(iP) = Vocs(iV).F(cPos);
    end
  end
  % CORRECT SPECTRAL LINE FOR NOISE
  for iP = 1:length(VocRange)
    switch iP
      case 1; Inds = [1,2];  %START
      case 2; Inds = [-1,1,2]; 
      case length(VocRange); Inds = -[1,2]; % END
      case length(VocRange)-1; Inds = [-2,-1,1];
      otherwise Inds = [-2,-1,1,2];  % MIDDLE
    end
    % AVERAGE OVER NEIGHBORS IF DEVIATION TOO STRONG
    cMean = nanmean(cSpecLine(iP+Inds));
    if abs(1  - cSpecLine(iP)/cMean ) > 0.03
      cSpecLine(iP) = cMean;
    end
  end

  % INTERPOLATE SPECTRAL LINE ACROSS 
  cSpecLine = interp1(find(~isnan(cSpecLine)),...
    cSpecLine(~isnan(cSpecLine)),...
    [1:length(cSpecLine)],'linear');
  
  Vocs(iV).SpecLine = cSpecLine;
  
  % RECOMPUTE SPECTRAL PURITY
  D = Vocs(iV).Spec{iLarger}; SD=zeros(size(D,2)-2,1); 
  for i=1:size(D,2)-2 
    cD = D(:,[ i : i+2 ]);
    iAll = find(cD); [iX,iY] = ind2sub(size(cD),iAll);
    SD(i) = diff(prctile(iX,[10,90]));
  end
  Vocs(iV).SpecPurityMean = nanmean(SD);
    % CENTER FREQUENCY
  Vocs(iV).FMean = nanmean(Vocs(iV).SpecLine);
  % MAXIMAL FREQUENCY
  Vocs(iV).FMax = prctile(Vocs(iV).SpecLine,97);
  % MINIMAL FREQUENCY
  Vocs(iV).FMin = prctile(Vocs(iV).SpecLine,3);
  % FREQUENCY RANGE
  Vocs(iV).FRange = Vocs(iV).FMax-Vocs(iV).FMin;
  % STARTING FREQUENCY
  Vocs(iV).FStart = mean(Vocs(iV).SpecLine(1:2));
  % STOPPING FREQUENCY
  Vocs(iV).FStop = mean(Vocs(iV).SpecLine(end-1:end));
    
  % VARIANCE 
  Vocs(iV).Variance = var(Vocs(iV).SpecLine);
  % VARIANCE OF SPECTRAL LINE
  Vocs(iV).SpecLineVar = var(diff(Vocs(iV).SpecLine));
  % LOCAL VARIABILITY
  Vocs(iV).LocalChange = sum(abs(diff(Vocs(iV).SpecLine)))/size(Vocs(iV).Spec,2);  
  
  % POSITION IN CURRENT PHRASE AND PHRASE NUMBER
  LocalPosition = find(IVIs(iV:-1:1)>0.2,1,'first');
  if isempty(LocalPosition) LocalPosition = 1; end
  Vocs(iV).LocalPosition = LocalPosition;
  Vocs(iV).PhraseNumber = PhraseNumberByVoc(iV);
  Vocs(iV).PhraseLength = PhraseLengthByVoc(iV);
  
  % ESTIMATE LOCATION OF THE VOCALIZATION
  if P.ComputeLocation
    
    TimeSteps =  [0:0.001:Vocs(iV).Duration];
    StartTime = Vocs(iV).StartTime;
    StopTime = Vocs(iV).PostTime;
    Range = 0.02; % s, before and after to analyze
    SR = Vocs(iV).SRSound;
    LocTmp = zeros(length(TimeSteps),2);
    for iPos=1:length(TimeSteps)
      RelTime  = TimeSteps(iPos);
      StartStep = round(SR * (StartTime + RelTime - Range));
      StopStep = round(SR * (StopTime + RelTime + Range));
      NSteps    = length(Vocs(iV).Sound{iM});
      Ind           = [max([1,StartStep]) : min([NSteps,StopStep]) ];
      for iM = 1:NMics
        LocSound{iM} = Vocs(iV).Sound{iM}(Ind);
      end
      LocTmp(iPos,1:2) = ...
        VocLocalizer('Sounds',LocSound,'SR',Vocs(iV).SRSound,...
        'CorrTime',P.CorrDur,'DelayRange',P.CorrDur,'HighPass',25000,...
        'CorrMethod',{'GCC','XCorr'},'CenterShift',P.CenterShift,...
        'MicrophonePositions',P.MicrophonePositions,'SourceHeight',P.SourceHeight,'FIG',0);
    end
    Vocs(iV).Location = mean(LocTmp,1);

  end
  
  % PLOT PROPERTIES FOR CHECKING
  if P.CheckProperties
    figure(1); clf; 
    set(gcf,'PaperSize',[8.5,10],'PaperPosition',[0,0,8.5,10]);
    AxisLabelOpt = {'FontSize',8};
    [DC,AH] = axesDivide(ones(1,NMics),[2,1],[0.16,0.12,0.8,0.8],'c');
    for iA=1:numel(AH) set(AH(iA),'FontSize',7); end
    TimeSound = ([1:length(Vocs(iV).Sound{1})]/Vocs(iV).SRSound - StartTime)*1000;
    TimeSpec = ([1:size(Vocs(iV).Spec{1},2)]/Vocs(iV).SRSpec - StartTime)*1000;
    for iS=1:NMics
      % VOLTAGE
      PreSteps = [1:round(StartTime * Vocs(iV).SRSound)];
      SD = std(Vocs(iV).Sound{iS}(PreSteps));
      axes(AH(2,iS)); hold on;
      plot(TimeSound,Vocs(iV).Sound{iS}/SD,'k');
      xlim([TimeSound([1,end])]);
      xlabel('Time [ms]',AxisLabelOpt{:});
      ylim([-10,10]);
      if iS==1; ylabel('Voltage [noise S.D.]',AxisLabelOpt{:}); end
      
      % SPECTROGRAM
      axes(AH(1,iS));  hold on; box on; 
      imagesc(TimeSpec,Vocs(iV).F/1000,Vocs(iV).Spec{iS}/SD); 
      set(gca,'YDir','normal');
      xlim([TimeSpec([1,end])]+[-5,5]);
      ylim([0,125]);
      Range = TimeSpec([1,end]);
      plot(Range,repmat(Vocs(iV).FMin/1000,[2,1]),'r');
      plot(Range,repmat(Vocs(iV).FMean/1000,[2,1]),'Color',[1,0.5,0],'LineWidth',2);
      plot(Range,repmat(Vocs(iV).FMax/1000,[2,1]),'r');
      plot(0,Vocs(iV).FStart/1000,'.','Color',[0,0,1],'MarkerSize',14);
      plot(Vocs(iV).Duration*1000,Vocs(iV).FStop/1000,'.','Color',[0.2,0.2,1],'MarkerSize',14);
      plot(TimeSpec(VocRange),Vocs(iV).SpecLine/1000,'b');
      if iS == 1; ylabel('Freq. [kHz]',AxisLabelOpt{:});  end
      title(['Microphone ',num2str(iS)],'FontSize',8,'FontWeight','bold');
      colormap(HF_colormap({[1,1,1],[0,0,0]},[0,1]));
      if iS ==1 text(0.02,0.04,[Vocs(iV).Animal,'r',num2str(Vocs(iV).Recording),'t',num2str(Vocs(iV).Start,4)],'FontSize',5,'Units','n'); end
    %  h = colorbar('Location','southoutside'); h.Label.String = 'Level [arb.]';
    end
    %     % HANDLING THE  HOME DIRECTORY IN WINDOWS AND UNIX R?mi 02/05/2016
    %     if ispc
    %       home_path=[getenv('HOMEDRIVE') getenv('HOMEPATH')];
    %     else
    %       home_path=getenv('HOME');
    %     end
    %     print(gcf,[home_path '/Desktop/LastVocalization.pdf'],'-dpdf')
    pause
  end
  
  if mod(iV,30)==0 fprintf('\n'); end
  
end

% SELECT BASED ON CERTAIN PROPERTIES
IndAll = logical(ones(size(Vocs)));
for iS = 1:length(P.Selector)
  Property = P.Selector{iS}{1};
  Condition = P.Selector{iS}{2};
  Values = [Vocs.(Property)];
  cInd = eval(['Values ',Condition]);
  IndAll = logical(IndAll.*cInd);
end
Vocs = Vocs(IndAll);

% COMPUTE PROPERTIES OF THE SET OF VOCALIZATIONS
% PCA based classification
%Vocs =  LF_pcaVoc(Vocs,P.Method,P.MaxDims,P.BaseData);
  
% COMBINE WITH OTHER INFORMATION
% Fuse with Animal Position Data
% Contexts :
%     Nose-to-Nose Distance, combined with Animal Distance
%     # of social interaction in the session


% PLOT RESULTS WITH DIFFERENT COLOR CODES
if P.FIG
  figure(P.FIG); clf;
  % Change Number of Graphs (min,max,'c')
  [DC,AH] = axesDivide(1,6,'c');
  % Change Parameters of the Number of Graphs
  % ColorScalers = {'Duration','FMean','AnimalNumber','FMin','Number'}; 
  ColorScalers = {'Duration','FMean','AnimalNumber','FMin','Number','Location'}; 
  CMap = hot(256); CMap = CMap(1:end-50,:);
  for iC=1:length(ColorScalers)
    cData = [Vocs.(ColorScalers{iC})];
    MIN = min(cData);
    cData = cData - MIN;
    Normalizer = max(cData);
    if Normalizer ==0 Normalizer = 1; end
    axes(AH(iC)); hold on;
    for iV=1:length(Vocs)
      if ~isnan(Vocs(iV).(ColorScalers{iC}))
        cColor = CMap(round((size(CMap,1)-1)*(Vocs(iV).(ColorScalers{iC})-MIN)/Normalizer)+ 1,:);
      else
        cColor = [0,0,0];
      end
      plot3(AH(iC),Vocs(iV).Projection(1),Vocs(iV).Projection(2),Vocs(iV).Projection(3),'.',...
        'Color',cColor,'MarkerSize',20,'ButtonDownFcn',{@LF_showVoc,Vocs,iV});
    end
    axis auto; view(35,10); box off; grid on;
    title(ColorScalers{iC});
    set(gca,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[])
    view(70,72)
  end
end

function LF_showVoc(H,E,Vocs,cV)
  
  Type =  get(gcf,'SelectionType');
  switch Type
    case 'normal'; Range = 2;
    case 'alt'; Range = 0;
  end
      
  figure(2001); clf
  [DC,AH] = axesDivide(1,3,[0.15,0.15,0.8,0.8],'c');
  for iV=cV-Range:cV+Range
    for iM=1:2
      axes(AH(iM)); hold on;
      if iV<1 || iV> length(Vocs) continue; end
      MAX(iM,iV) = max(Vocs(iV).Spec{iM}(:));
      Time = Vocs(iV).Start + [1:size(Vocs(iV).Spec{iM},2)]/Vocs(iV).SRSpec;
      if Range == 0 Time = Time - Time(1); end 
      imagesc(...
        Time,...
        Vocs(iV).F/1000,...
        Vocs(iV).Spec{iM}/MAX(iM,iV));
      plot([Time(1),Time(1)],[0,125],'g');
      plot([Time(end),Time(end)],[0,125],'r');
      set(gca,'YDir','normal'); axis tight;
      if Range == 0 xlim([Time(1),Time(1)+ 0.1]); end
    
      axes(AH(3)); hold on;
      plot(Time,Vocs(iV).SpecLine,'k');
    end
    axis tight;
    ylim([0,125]);
    if Range == 0 xlim([Time(1),Time(1)+ 0.1]); end
  end
  set(AH(1:2),'clim',[0,0.5]);
  String = ['#',num2str(cV),' ',Vocs(iV).Animal,' R',num2str(Vocs(cV).Recording)];
  if isfield(Vocs(cV),'AnimalPartner') String = [String,' with ',Vocs(cV).AnimalPartner]; end
  set(gcf,'Name',String);

  CM  = colormap('gray'); colormap(flipud(CM));

function Vocs =  LF_pcaVoc(Vocs,Method,MaxDims,BaseData)

  X = zeros(length(Vocs),round(max([Vocs.Duration])*Vocs(1).SRSpec));
  % PREPARE DATA MATRIX
  if ~isempty(BaseData)
    X = [];
    for iF = 1:length(BaseData)
      cField = BaseData{iF};
      cData = [Vocs.(cField)]
      if size(cData,1)~=length(Vocs) cData = cData'; end
      X = [X,cData];
    end
  else
    for iV = 1:length(Vocs)
      tmp = Vocs(iV).SpecLine;
      X(iV,1:length(tmp)) = tmp;
    end
  end
  
  X = zscore(X);
  
  % RUN DIMENSIONALITY REDUCTION
  switch Method
    case 'tSNE'; Pars = {30,30};
    otherwise Pars = {};
  end

  switch Method
    case 'PCA';
      [C,XR,Variances] = princomp(X);
      fprintf('%f %f %f [%%]\n',Variances(1:3)/sum(Variances));
    otherwise
      XR = compute_mapping(X,Method,MaxDims);
  end

  % CLASSIFICATION INTO DIFFERENT CATEGORIES
  % use hierarchical clustering like in spike soting
   D = pdist(XR(:,1:MaxDims));
   Z = linkage(D);
   Classes = cluster(Z,'maxclust',10);
  
  % ASSIGN PROPERTIES BACK TO THE VOCALIZATION
  for iV = 1:length(Vocs)
    Vocs(iV).Class = Classes(iV);
    Vocs(iV).Projection = XR(iV,1:MaxDims);
  end    
  
function [XX,MidDist] = LF_xcorr(V1,V2,Dur,SR)  

  CorrSteps = round(Dur * SR);
  XX = xcorr(V1,V2,CorrSteps,'unbiased');
  % NOTE : Negative Position of the Peak, means first argument is leading
  XX = abs(hilbert(XX));
  [M,Pos] = max(XX);
  DeltaTime = (Pos-(CorrSteps+1))/SR;
  
  % COMPUTE POSITION BASED ON SETUP GEOMETRY AND CORRELATION
  MicDist = 0.46; %m % Measurements from new setup
  % Estimate of previous distances : 0.38
  MicHeight = 0.354; %m % Measurements from new setup
  % Estimate of previous height: 0.35
  VSound = 340.29; %m/s
  % DeltatTme is negative, if signal arrives first on the left (channel 1), i.e. position is also on the left
  DeltaDist = VSound * DeltaTime;
  D = DeltaDist;
  H = MicHeight;
  S  = MicDist;
  MidDist = (D*sqrt(D^2-4*H^2-S^2))/sqrt(4*D^2-4*S^2);
