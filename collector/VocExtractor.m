function Data = VocExtractor(varargin)
% COLLECT PROPERTIES AND PERFORM INITIAL ANALYSIS OF VOCALIZATIONS  

P = parsePairs(varargin);
checkField(P,'Data')
checkField(P,'Range');
checkField(P,'SR',250000)
checkField(P,'Channels',[1,2]);
checkField(P,'HighPass',50000);
checkField(P,'LowPass',90000);
checkField(P,'FIG',1); % 

% EXTRACT DATA
Ind = find((P.Data.AnalogIn.Data.Data.Time>P.Range(1)).*(P.Data.AnalogIn.Data.Data.Time<P.Range(2)));
Data = P.Data.AnalogIn.Data.Data.Analog(Ind,P.Channels);


% POTENTIALLY FILTER THE DATA
if P.HighPass
  [b,a]=butter(8,P.HighPass/(P.SR/2),'high');
  for i=1:length(P.Channels) 
    Data(:,i) = filter(b,a,Data(:,i)); 
  end
end

if P.LowPass
  [b,a]=butter(8,P.LowPass/(P.SR/2),'low');
  for i=1:length(P.Channels)
    Data(:,i) = filter(b,a,Data(:,i));
  end
end

