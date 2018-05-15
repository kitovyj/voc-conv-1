function Vocs = SC_VocAnalyzer(varargin)  
  
P = parsePairs(varargin);
checkField(P,'VocJitter',0.002);
checkField(P,'FWindow',0.001);
checkField(P,'Selection','GapTest');
checkField(P);

P.Selected = C_RecordingSets('Project','USVLoc','Set',P.Selection);
switch P.Selection
  case {'Gap','GapTest'};
    Paradigm = 'Interaction';
    P.MicrophonePositions = {[-0.23,0,0.314],[0.23,0,0.314]};
    P.SourceHeight = 0;
    P.CenterShift = 0.009;
  case 'Platform1D';
    P.MicrophonePositions = {[-0.296,0.279,0.383],[0.296,0.279,0.383]};
    P.SourceHeight = 0.25;
    P.CenterShift = 0;
    Paradigm = 'AudioVideoRecording';
  otherwise error('Selection is not defined.');
end
  
for iR=1:length(P.Selected)
  cRecord = P.Selected{iR};
  Vocs = VocCollector('Animals',cRecord(1),'Recording',cRecord{2},'Paradigm',Paradigm,...
    'Reload',1,'VocJitter',P.VocJitter,'PreTime',0.01,'PostTime',0.01);
  [Path,Paths] = C_getDir('Animal',cRecord{1},'Recording',cRecord{2});
  ResPath = [Paths.DB,'Results'];
  try mkdir(ResPath); end;
  VocPath = [ResPath,filesep,'Vocalizations'];
  try mkdir(VocPath); end;
  VocFile = [VocPath,filesep,'Vocalizations.mat'];
  Vocs = VocAnalyzer(Vocs,'FIG',0,'CenterShift',P.CenterShift,...
    'MicrophonePositions',P.MicrophonePositions,'SourceHeight',P.SourceHeight);
  save(VocFile,'Vocs');
end
