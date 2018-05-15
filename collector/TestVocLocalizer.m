function TestVocLocalizer(varargin)
% TEST VOCALIZATION LOCALIZATION

P = parsePairs(varargin);
% DEFINE STIMULUS
checkField(P,'Stimulus','Noise');
checkField(P,'StimulusPosition',[-0.02,0,0]); % in Meters
checkField(P,'Duration',1); % [s] : Duration of the stimulus
checkField(P,'FBase',40000); % Base frequency of the stimulus
checkField(P,'DeltaX',0.3); % Offset in octaves of the stimulus
% DEFINE RECORDING CONFIGURATION (SPACE)
checkField(P,'MicrophonePositions',{[-0.23,0,0.314],[0.23,0,0.314]}); % in Meters
checkField(P,'SR',250000);
% ADD NOISE TO STIMULUS
checkField(P,'Signal2Noise',10);
checkField(P,'NoiseTime',0.01);
checkField(P,'Seed',[]);
% DEFINE ESTIMATION METHOD
checkField(P,'CorrMethod','GCC'); 
checkField(P,'FIG',1); 
checkField(P);

% INITIALIZE NOISE
if isempty(P.Seed) P.Seed = ceil(2^32*rand); end
RR = RandStream('mt19937ar','Seed',P.Seed);

% CREATE STIMULUS SIGNAL
T = [0:1/P.SR:P.Duration];
TSteps = length(T);
switch P.Stimulus
  case 'Noise';
    Stimulus = randn(TSteps,1);
  case 'Vocalization';
    XUp = linspace(0,P.DeltaX,TSteps/5)';
    XDown = linspace(P.DeltaX,0,TSteps/5)';
    XMiddle = repmat(P.DeltaX,round([TSteps-2*TSteps/5,1]));
    X=[XUp;XMiddle;XDown];
    F = P.FBase.*2.^X;
    phaseinc = 1./P.SR.*F;
    phases = cumsum(phaseinc);
    Stimulus = sin(2*pi*phases);
  otherwise error(['Stimulus ',P.Stimulus,' not specified.']);
end
Stimulus = Stimulus / std(Stimulus);

% COMPUTE TIME SHIFTS BASED ON MICROPHONE GEOMETRY
NMics = length(P.MicrophonePositions); VSound = 343; %m/s
for iM = 1:NMics 
  Distances(iM) = norm(P.MicrophonePositions{iM} - P.StimulusPosition);
  Shifts(iM) = Distances(iM)/VSound; % in seconds;
end

% CREATE TIME SHIFTED SIGNALS WITH NOISE
for iM=1:NMics
  NSteps = round(P.SR * (P.Duration + P.NoiseTime));
  Sounds{iM} = zeros(NSteps,1);
  cShiftSteps = round(P.SR*Shifts(iM));
  Sounds{iM}(cShiftSteps+1:cShiftSteps + length(Stimulus)) = Stimulus;
  Sounds{iM} = P.Signal2Noise*Sounds{iM} + RR.randn(NSteps,1);
  Time = [1:250]/P.SR;
  IIR{iM} = Time.*exp(-Time/0.00005).*sin(2*pi*(50000+iM*10000)*Time); 
  Sounds{iM} = conv(Sounds{iM},IIR{iM});
end

% REESTIMATE LOCATION FROM THE SOUNDS
CorrTime = 2*max(Shifts);
[MidDist,SCorr,CorrTime,DeltaTime,CorrMidDist] = VocLocalizer('Sounds',Sounds,'SR',P.SR,...
  'CorrMethod',P.CorrMethod,'CenterShift',0,'CorrTime',CorrTime,'DelayRange',CorrTime,...
  'MicrophonePositions',P.MicrophonePositions,'StimulusPosition',P.StimulusPosition,'LocMethod','Linear','FIG',P.FIG);