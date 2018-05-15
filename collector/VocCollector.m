function Vocs = VocCollector(varargin)
% 
% Example Usage:
%  Vocs = VocCollector('Animals',{'mouse3'},'Recording',80);
% 
%  Vocs = VocAnalyzer(Vocs);
  
P = parsePairs(varargin);
checkField(P,'DataSource','Controller'); % CAN BE EITHER Controller, or WAV
% CONTROLLER PARAMETERS
checkField(P,'Paradigm','');
checkField(P,'Animals',{});
checkField(P,'Recording',[]);
% WAV PARAMETERS
checkField(P,'Filename',[]);
if strcmp(P.DataSource,'WAV') & isempty(P.Filename) 
  error('Filename has to be provided if DataSource is WAV'); end;
% EXTRACTION PARAMETERS
checkField(P,'VocJitter',0.002);
checkField(P,'FWindow',0.001);
checkField(P,'FRange',[10000,125000]);
checkField(P,'PreTime',0);
checkField(P,'PostTime',0);
checkField(P,'Channels',[1,2])
checkField(P,'Reload',1)
checkField(P,'SRAI',[]);
checkField(P);

switch P.DataSource
  case 'Controller';
    % SELECT RECORDINGS FROM DATA BASE
    R = [];
    C_checkDatabase('Force',1);
    % GET ANIMAL IDs
    RA = mysql('SELECT * FROM subjects');
    for i=1:length(RA) IDsByAnimal.(RA(i).name) =RA(i).id; end
    if isempty(P.Animals) P.Animals = setdiff({RA.name},'test'); end
    
    % GET RECORDING INFO
    for iA=1:length(P.Animals)
      MYSQL = ['SELECT * FROM recordings WHERE '...
        'animal=''',P.Animals{iA},''' AND bad=0  '];
      if ~isempty(P.Recording)
        MYSQL = [MYSQL,' AND recording=',num2str(P.Recording)];
      end
      if ~isempty(P.Paradigm)
        MYSQL = [MYSQL,' AND paradigm=''',P.Paradigm,''''];
      end
      R = [R;mysql(MYSQL)];
    end
    
    fprintf(['\n= = = = Found [ ',num2str(length(R)),' ] Recordings = = = =\n\n']);
    % MAKE RECORDINGS LOCAL
    for iR=1:length(R)
      Path = C_makeLocal('Animal',R(iR).animal,'Recording',R(iR).recording,...
        'Modules',{'AnalogIn','NIDAQ'});
    end
    
  case 'WAV'; % DEFINE RECORDINGS
    R.Filename = P.Filename; % Just one filename possible
    
  case 'Direct'; % 
    R.Filename = 1;
    
end
    
k=0; Vocs = [];
for iR=1:length(R)
  
  % LOAD DATA VOCALIZATIONS
  global CurrentData
  if isempty(CurrentData) || P.Reload || length(R)>1
    switch P.DataSource
      case 'Controller'; 
        fprintf(['\n= = = = = =  Recording : [ ',R(iR).animal,' R',num2str(R(iR).recording),' ] = = = = = = = = = = = =\n']);
        CurrentData = C_loadRecording('Animal',R(iR).animal,'Recording',R(iR).recording,...
          'Modules',{'AnalogIn','NIDAQ'});
          SRAI = CurrentData.General.Parameters.Setup.Audio.SRAI;

      case 'WAV';
        fprintf(['\n= = = = = =  Recording : [ ',escapeMasker(P.Filename),' ] = = = = = = = = = = = =\n']);
        [Data,SRAI] = audioread(P.Filename);
        CurrentData.AnalogIn.Data(1).Data.Analog = Data;
        CurrentData.AnalogIn.Data(1).Data.Time = 0;
        P.Channels = intersect(P.Channels,[1:size(Data,2)]);        
        
      case 'Direct'
        CurrentData.AnalogIn.Data(1).Data.Analog = P.Data;
        CurrentData.AnalogIn.Data(1).Data.Time = 0;
        P.Channels = [1:size(P.Data,2)];
        SRAI = P.SRAI;
        
      otherwise error('Data Source not implemented.');
        
    end
    
    Reloaded = 1;
  else Reloaded = 0;
  end
  
  CurrentData.AnalogIn.Data(1).Data.Analog = CurrentData.AnalogIn.Data(1).Data.Analog(:,P.Channels);
  NChannels = size(CurrentData.AnalogIn.Data(1).Data.Analog,2);

  % DOWN SAMPLE FOR TOO HIGH SAMPLE RATES
  if SRAI == 500000 && Reloaded
    CurrentData.AnalogIn.Data(1).Data.Analog =  CurrentData.AnalogIn.Data(1).Data.Analog(1:2:end,:);
    CurrentData.AnalogIn.Data(1).Data.Time =  CurrentData.AnalogIn.Data(1).Data.Time(1:2:end);
    SRAI = 250000;
  end
  
  % ADAPT UPPER FREQUENCY END
  %P.FRange(2) = SRAI/2;
  
  % EXTRACT VOCALIZATIONS
  RecordingOnset = CurrentData.AnalogIn.Data(1).Data.Time(1);
  FWindowSteps = P.FWindow*SRAI;
  FWindowSteps = 500;
  if NChannels > 0 % DATA NOT MISSING
    for iT=1:length(CurrentData.AnalogIn.Data) % LOOP OVER TRIALS
      fprintf(['    Trial ',n2s(iT),'\n']);
      for iM=1:NChannels % LOOP OVER MICROPHONES
        fprintf(['    > Computing Spectrogram on Channel ',num2str(iM),'\n']);
        % SET PARAMETERS
        [S{iM},F,T,Thresh(iM),SRSpec] = HF_specgram(CurrentData.AnalogIn.Data(iT).Data.Analog(:,iM),...
          FWindowSteps,SRAI,P.FRange,FWindowSteps/2,1,1);
        TVocs{iM} = LF_findVocs(S{iM},F,T,'Threshold',Thresh(iM),'Channel',iM);
      end
      
      if ~isempty(cell2mat(TVocs))
        if NChannels == 1; TVocsAll = TVocs{1}; InterVocTimes = [NaN,diff(TVocsAll(1,:))];
        else  [TVocsAll,InterVocTimes] = LF_fuseVocTimes(TVocs,P.VocJitter,SRSpec);
        end
        
        [cVocsSound,cVocsSpec,TVocsAll] = LF_getVocs(iT,S,TVocsAll,SRAI,SRSpec,P.PreTime,P.PostTime);
        
        if ~isempty(TVocsAll)
          for iV=1:size(TVocsAll,2) % LOOP OVER FOUND VOCALIZATIONS
            k=k+1;
            for iM=1:NChannels % LOOP OVER MICROPHONES
              % COLLECT IN AN INTERMEDIATE FORMAT
              Vocs(k).Sound{iM} = cVocsSound{iM}{iV};
              Vocs(k).Spec{iM} = full(abs(cVocsSpec{iM}{iV}));
              Vocs(k).SpecPurity{iM} = LF_SpecPurity(Vocs(k).Spec{iM});
              switch P.DataSource
                case 'Controller';
                  Vocs(k).Recording = R(iR).recording;
                  Vocs(k).Animal = R(iR).animal;
                  Vocs(k).AnimalNumber = str2num(R(iR).animal(6:end));
                  switch P.Paradigm
                    case 'Interaction';
                      if isfield( CurrentData.General.Paradigm.Parameters,'InteractionPartner')
                        Vocs(k).AnimalPartner = CurrentData.General.Paradigm.Parameters.InteractionPartner;
                        Vocs(k).AnimalPartnerNumber = str2num(Vocs(k).AnimalPartner);
                      end
                  end
                case 'WAV';
                  Vocs(k).Filename = R(iR).Filename;
                  Vocs(k).Animal = 'Unknown'; Vocs(k).Recording = 0;
              end         
              Vocs(k).Trial = iT;
              Vocs(k).Number = iV;
              % START AND STOP OF CUTOUT DATA
              Vocs(k).StartWin = TVocsAll(1,iV) + RecordingOnset;
              Vocs(k).StopWin = TVocsAll(2,iV) + RecordingOnset;
              Vocs(k).DurationWin = Vocs(k).StopWin - Vocs(k).StartWin;
              % START AND STOP OF ACTUAL VOCALIZATION
              Vocs(k).Start = TVocsAll(1,iV)+P.PreTime + RecordingOnset;
              Vocs(k).Stop = TVocsAll(2,iV)- P.PostTime + RecordingOnset;
              Vocs(k).Duration = Vocs(k).Stop - Vocs(k).Start;
              
              Vocs(k).PreTime = P.PreTime;
              Vocs(k).PostTime = P.PostTime;
              Vocs(k).Interval = InterVocTimes(iV);
              Vocs(k).SRSound = SRAI;
              Vocs(k).SRSpec = SRSpec;
              Vocs(k).dF = F(2)-F(1);
              Vocs(k).F = F;
              Vocs(k).Time = Vocs(k).StartWin + [1:size(Vocs(k).Spec{iM},2)]/Vocs(k).SRSpec;
            end
          end
        end
      end
    end
  end
end

% COMPUTE SPECTROGRAM WITH FREQUENCY RANGE REDUCTION
function [Data,F,T,Thresh,SRSpec] = LF_specgram(Data,Nfft,SRAI,FRange)

  SpecSteps = floor(length(Data)/Nfft*2);
  Data = reshape(Data(1:SpecSteps*Nfft/2),Nfft/2,SpecSteps);
  Data = [Data(:,1:end-1) ; Data(:,2:end)];
  SpecSteps = SpecSteps - 1;
  SRSpec = SRAI/(Nfft/2);
  
  % WINDOW DATA
  Window = hanning(Nfft);
  Data = bsxfun(@times,Data,Window);
  
  % Fourier transform
  Data = abs(fft(Data,Nfft));
  
  % COMPUTE THE FREQUENCY AND TIME VECTORS
  F = linspace(0,SRAI/2,Nfft/2+1);
  T = [0:1/SRSpec:(SpecSteps-1)/SRSpec];
  
  % SELECT RANGE OF FREQUENCIES
  Ind = logical((F>=FRange(1)) .* (F<=FRange(2)));
  F = F(Ind);
  
  % Assume input signal is real, and take only the necessary part
  Data = Data(Ind,:);
  
  % Convert to sparse representation based on threshold 
  ThreshData = Data(end/2,end-20000:end-10000);
  Thresh = median(ThreshData) + 2*std(ThreshData);
  Ind = find(abs(Data) >= Thresh);
  [Ind1,Ind2] = ind2sub(size(Data),Ind);
  Data= sparse(Ind1,Ind2,double(Data(Ind)),size(Data,1),size(Data,2));
  
% FIND THE VOCALIZATIONS
function TVocs = LF_findVocs(S,F,T,varargin)
    
    P = parsePairs(varargin);
    checkField(P,'Threshold')
    checkField(P,'DurationMin',0.007);
    checkField(P,'DurationMax',0.5);
    checkField(P,'FreqThreshMean',25000);
    checkField(P,'PurityThresh','auto');
    checkField(P,'SpecDiscThresh',0.8);
    %checkField(P,'SpecDiscThresh',0.7);
    %checkField(P,'MergeClose',0.005);
    checkField(P,'MergeClose',0.015);
    checkField(P,'Log',0);
    checkField(P,'Channel',NaN);
    checkField(P,'FilterDuration',0.01);
    
    fprintf(['\tAnalyzing Channel ',n2s(P.Channel),'  : ']);
    
    
    dT = diff(T(1:2)); dF = diff(F(1:2));
    FilterSteps = round(P.FilterDuration/dT);
   
    % CONVERT SPECGRAM TO SPECTRAL POWER  
    S = sparse(medfilt2(full(S),[1,5]));
    [i,j,s] = find(S);   SPow = S.^2;
    % LOG SCALE
    if P.Log;   SPow = sparse(i,j,log(S/P.Threshold),size(S,1),size(S,2));   end
    
    % COMPUTE SPECTRAL PURITY
    totPower = sum(SPow);   [maxPower,maxIndx] = max(SPow);
    iNonZero = find(totPower);
    SpecPurity = zeros(size(T));
    SpecPurity(iNonZero) = maxPower(iNonZero)./totPower(iNonZero);

    if isequal(P.PurityThresh,'auto')
      [H,B] = hist(SpecPurity,[100]);
      [~,Ind] = max(H); BasePurity = B(Ind);
      Ind = find(SpecPurity<BasePurity);
      SD = sqrt(sum(SpecPurity(Ind) - BasePurity).^2)/length(Ind);
      P.PurityThresh = BasePurity + 2*SD;
    end

%     if P.FilterDuration
%       SpecPurity = medfilt1(SpecPurity,FilterSteps);
%     end
    
    % SELECT VOCALIZATIONS
    iBad = SpecPurity <= P.PurityThresh;
    if P.FreqThreshMean;  
      spfreq = sparse(i,j,i,size(S,1),size(S,2));
      MeanFreq = sum(spfreq.*SPow)*dF;
      MeanFreq(iNonZero) = MeanFreq(iNonZero)./totPower(iNonZero);
      if P.FilterDuration
        MeanFreq = medfilt1(full(MeanFreq),FilterSteps);
      end
      iBad(end+1,:) = MeanFreq < P.FreqThreshMean;  
    end
    if P.SpecDiscThresh    
      SD = specdiscont(SPow);
      SD = medfilt1(SD,FilterSteps);
      iBad(end+1,:) = SD > P.SpecDiscThresh; 
    end
    
    % FIND INDICES THAT FULLFILL ALL CRITERIA
    iBad = find(max(iBad,[],1));
    DurationMinSteps = P.DurationMin/dT;      iMin = find(diff(iBad) > DurationMinSteps);
    DurationMaxSteps = P.DurationMax/dT;     iMax = find(diff(iBad) < DurationMaxSteps);   
    iLength = intersect(iMin,iMax);
    TVocs = [T(iBad(iLength)); T(iBad(iLength+1))];

    fprintf([' ',num2str(size(TVocs,2)),' ==(merge close)==>']);

    % MERGE CLOSE PAIRS
    if P.MergeClose
      DeltaT = TVocs(1,2:end)-TVocs(2,1:end-1);
      iClose = find(DeltaT <  P.MergeClose);
      for i = length(iClose):-1:1
        TVocs(2,iClose(i)) = TVocs(2,iClose(i)+1);
        TVocs(:,iClose(i)+1) = [];
      end
    end
    
    fprintf([' ',num2str(size(TVocs,2)),'\n']);
   
    
% FUSE THE VOCALIZATION TIMES ACROSS THE CHANNELS
function [TVocsAll,IVI] = LF_fuseVocTimes(TVocs,VocJitter,SRSpec)
  % FUSE THE WHISTLES ACROSS THE TWO SIDES
  fprintf('\tFusing Vocalizations  :  ');
  
  % CONSTRUCT COMMON VECTOR BETWEEN
  NChannels = length(TVocs);    
  MaxDur = max([TVocs{1}(:);TVocs{2}(:)]);
  MaxSteps = ceil(MaxDur*SRSpec);
  iGood = zeros(NChannels,MaxSteps);
  for iM=1:NChannels
    for iV = 1:size(TVocs{iM},2)
      cInd = [round(TVocs{iM}(1,iV)*SRSpec) : round(TVocs{iM}(2,iV)*SRSpec)];
      iGood(iM,cInd) = 1;
    end
  end
  
  % EXTRACT THE OVERLAPPING VOCALIZATIONS
  iFuse = [-1,find(sum(iGood,1)),MaxSteps+2];
  StopInd = find(diff(iFuse)>1);
  StopTimes = iFuse(StopInd(2:end))/SRSpec;
  StartTimes = iFuse(StopInd(1:end-1)+1)/SRSpec;
  TVocsAll = [StartTimes;StopTimes];
  IVI = [NaN,diff(TVocsAll(1,:))];
  NVocsAll = size(TVocsAll,2);
  fprintf([num2str(NVocsAll),'\n']);
  
% EXTRACT THE VOCALIZATIONS BASED ON THE TIMES
function [VSound,VSpec,TVocsAll] = LF_getVocs(iT,S,TVocsAll,SRSound,SRSpec,PreTime,PostTime)

  global CurrentData;
  
  NChannels = length(S);
  
  for iV = 1:size(TVocsAll,2)
    cTimes = TVocsAll(:,iV);
    cInd = round(cTimes*SRSpec);
    Started = 0;
    while ~Started
      for iC = 1:NChannels;   Started = abs(Started + S{iC}(:,cInd(1)));    end
      if ~Started cInd(1) = cInd(1)+1; end
    end
    Stopped = 0;
    while ~Stopped
      for iC = 1:NChannels;   Stopped = abs(Stopped + S{iC}(:,cInd(1)));    end
      if ~Stopped cInd(2) = cInd(2)-1; end
    end
    cTimes = cInd/SRSpec + [-PreTime;PostTime];
    TVocsAll(:,iV) = cTimes;
    cIndSpec = round(cTimes*SRSpec);
    cIndSound = round(cTimes*SRSound);
    for iM=1:NChannels
      % COLLECT SPECTROGRAM
      VSpec{iM}{iV} = S{iM}(:,cIndSpec(1):cIndSpec(2));
      
      % COLLECT SOUND PRESSURE
      VSound{iM}{iV} = double(CurrentData.AnalogIn.Data(iT).Data.Analog(cIndSound(1):cIndSound(2),iM));      
    end
  end
  
  function SpecPurity = LF_SpecPurity(SPow)
      totPower = sum(SPow);  
   [maxPower,maxIndx] = max(SPow);
    iNonZero = find(totPower);
    SpecPurity = zeros([size(SPow,2),1]);
    SpecPurity(iNonZero) = maxPower(iNonZero)./totPower(iNonZero);
    
 