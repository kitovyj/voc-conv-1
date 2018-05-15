function [Result,Stimulus] = ReplaySounds(varargin)

P = parsePairs(varargin);
checkField(P,'Stimulus','Noise');
checkField(P,'Duration',2);
checkField(P,'Repetitions',1);
checkField(P,'NSweeps',10);
checkField(P,'Pause',0.1);
checkField(P,'FBase',30000);
checkField(P,'DeltaX',0.3);
checkField(P,'Calibrate',1);
checkField(P,'Scale',20);
checkField(P,'Speaker','FostexT250D');
checkField(P,'Microphone','AvisoftCM16');
checkField(P,'SR',250000);

% CREATE STIMULUS
Stimulus = LF_createStimulus(P);
Stimulus = [Stimulus;zeros(1000,1)];
fprintf(['Created Stimulus : Total length : ',num2str(length(Stimulus)/P.SR),'s\n']);

% INITIALIZE SESSION
S = daq.createSession('ni');
S.addAnalogOutputChannel('Dev2',0,'Voltage');
for i=1:2
  S.addAnalogInputChannel('Dev2',i-1,'Voltage');
  S.Channels(i+1).TerminalConfig = 'SingleEnded';
end
S.Rate = P.SR;

% LOAD SPEAKER CALIBRATION
if P.Calibrate
  Sep = filesep;
  Path = which('Controller'); Path = Path(1:find(Path==Sep,1,'last'));
  Path = [Path,'Modules',Sep,'AudioOut',Sep,'Speakers',Sep];
  FileName = [Path,'SpeakerCalibration_',P.Speaker,'_',P.Microphone,'.mat'];
  SC = load(FileName); SC = SC.R;
  
  if P.SR~=SC.SR; error('Sampling Rates not Matched!'); end
  Stimulus = P.Scale*conv(Stimulus,SC.IIR60dB);
end

S.queueOutputData(repmat(Stimulus,P.Repetitions,1));
Result = S.startForeground;
if P.Calibrate
  DelaySteps = round(SC.ConvDelay*P.SR);
  Result = Result(DelaySteps:end,:);
  Stimulus = Stimulus(1:size(Result,1));
end
clear S;

% PREPARE STIMULUS
function Stimulus = LF_createStimulus(P);

switch P.Stimulus
  case 'Noise';
    Stimulus = randn(P.SR*P.Duration,1);
  case 'FM';
    Stimulus = []; cPause = zeros(P.Pause*P.SR,1);
    T = [0:1/P.SR:P.Duration];
    X = linspace(0,P.DeltaX,length(T))';
    for i=1:P.NSweeps
      F = (1+i/P.NSweeps)*P.FBase.*2.^X;
      phaseinc = 1./P.SR.*F;
      phases = cumsum(phaseinc);
      cS = sin(2*pi*phases);
      Stimulus = [Stimulus;cPause;cS];
    end
  case 'UpDown';
    Stimulus = []; cPause = zeros(P.Pause*P.SR,1);
    T = [0:1/P.SR:P.Duration];
    X = linspace(0,P.DeltaX,length(T)/2)';
    X=[X;flipud(X)];
    for i=1:P.NSweeps
      F = (1+i/P.NSweeps)*P.FBase.*2.^X;
      phaseinc = 1./P.SR.*F;
      phases = cumsum(phaseinc);
      cS = sin(2*pi*phases);
      Stimulus = [Stimulus;cPause;cS];
    end
    
  case 'Vocal';
    Stimulus = []; cPause = zeros(P.Pause*P.SR,1);
    T = [0:1/P.SR:P.Duration];
    XUp = linspace(0,P.DeltaX,length(T)/5)';
    XDown = linspace(P.DeltaX,0,length(T)/5)';
    XMiddle = repmat(P.DeltaX,round([length(T)-2*length(T)/5,1]));
    X=[XUp;XMiddle;XDown];
    for i=1:P.NSweeps
      F = (1+i/P.NSweeps)*P.FBase.*2.^X;
      phaseinc = 1./P.SR.*F;
      phases = cumsum(phaseinc);
      cS = sin(2*pi*phases);
      Stimulus = [Stimulus;cPause;cS];
    end
    
  otherwise fprintf('Stimulus not known!');
end
