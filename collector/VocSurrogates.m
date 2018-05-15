function Vocs = VocSurrogates(varargin)
% COLLECT PROPERTIES AND PERFORM INITIAL ANALYSIS OF VOCALIZATIONS  

P = parsePairs(varargin);
checkField(P,'SR',250000)
checkField(P,'CorrTimes',[-0.0001:0.00001:0.0001]); % 0.1 ms
checkField(P,'Length',0.05); 
checkField(P,'Range',0.5);
checkField(P,'FBase',50000);
checkField(P,'CenterShift',0.01); %10mm of CenterShift based on measured discrepancy between position of left and right microphone
checkField(P,'Shapes',{'Descending'});
checkField(P,'DelayRange',0.000075); %75us
checkField(P,'Noise',0)
checkField(P,'FIG',1); % 


dt = 1/P.SR;
T = [0:1/P.SR:P.Length]'; 
NSteps = length(T);
x =  linspace(0,P.Range,NSteps)';  
f = P.FBase*2.^x;  

phaseinc = dt.*f;
phases = cumsum(phaseinc); 
STIM = sin(2*pi*phases);

for iV=1:length(P.CorrTimes)
  Vocs(iV).Sound{1} = P.Noise*randn(NSteps,1);
  Vocs(iV).Sound{2} = P.Noise*randn(NSteps,1);
  CorrSteps = round(P.CorrTimes(iV)*P.SR);
  Vocs(iV).Sound{1} = Vocs(iV).Sound{1} + STIM;
  if CorrSteps>0
    Vocs(iV).Sound{2}(CorrSteps+1:end) = Vocs(iV).Sound{2}(CorrSteps+1:end) + STIM(1:end-CorrSteps);
  elseif CorrSteps < 0
    Vocs(iV).Sound{2}(1:end+CorrSteps) = Vocs(iV).Sound{2}(1:end+CorrSteps) + STIM(-CorrSteps+1:end);
  else
    Vocs(iV).Sound{2} = Vocs(iV).Sound{2} + STIM;
  end
end



