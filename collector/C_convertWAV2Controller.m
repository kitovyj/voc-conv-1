function R = C_convertWAV2Controller(varargin)

P = parsePairs(varargin);
checkField(P,'Data');
checkField(P,'SR');

% RECREATE BASIC RECORDING INFORMATION
R.General.Parameters.General = struct('Animal','unknown','Recording',0,'Identifier','unknown_R0');
R.General.Parameters.Setup.Audio.SRAI = P.SR;

% RECREATE ANALOG IN STRUCTURE
R.AnalogIn.Trials = 1;
cTime = [0:length(P.Data)-1]'/P.SR;
R.AnalogIn.Data.Data = struct('Analog',vertical(P.Data),'Time',cTime);
R.AnalogIn.FromWAV = 1;

% RECREATE NIDAQ STRUCTURE
R.NIDAQ.Trials = 1;
NSteps = length(P.Data);
SRDAQ = 10000;
cTime = [0:round(cTime(end)*SRDAQ)]'/SRDAQ;
cData = ones(size(cTime)); cData([1,end]) = 0; 
R.NIDAQ.Data.Data = struct('Analog',cData,'Time',cTime);
R.NIDAQ.Data.CameraStartTime = [];
R.NIDAQ.Data.TrialStartTime = cTime(2);
R.NIDAQ.Data.TrialStopTime = cTime(end-1);

% RECREATE PARADIGM
R.General.Paradigm.HW.Inputs = struct('Type','NI','TypeID',1,'Channel','ai8','Name','Trial','Index',1,'Save',1);
R.General.Paradigm.HW.AnimalSensorsPhys = [];