function MultiViewer(varargin)
% Data Viewer for Multimodal Data Collected with Controller
% Note: At this point MultiViewer only works with consecutive trials!
% 
% Usage: 
%  MultiViever('Argument',Value,...);
%  Data                : MultiModel data in Controller format (obtained from C_loadRecording)
%  Transpose [0]       : Transpose video (Rotate by 90 degree);
%  MirrorX [0]         : Mirror video horizontally
%  MirrorY [0]         : Mirror video vertically
%  CMin [0]            : Minimum of color range
%  CMax [255]          : Maximum of color range
%  TrueLimits [0]      : Compute true color range limits (slower startup)
%  BackRange [ ]       : Averaging range for background subtraction
%  StartFrame [1]      : Frame to start on
%  CurvesByFrame [ ]   : Curves to display for each frame 
%                        (for adding annotations)
%  FIG [1]             : Number of figure to plot into. 
% 
% Keyboard Shortcuts :
% 
% Playing: 
% - Arrow left :  one frame back
% - Arrow right :  one frame forward
%
% Visual: 
% - g : show grid
% - +/- : change color range on the spectrogram 
% - S : Switch speaker
% Author : benglitz@gmail.com

%% PROCESS ARGUMENTS
P =  parsePairs(varargin);
checkField(P,'Data',[]);
if isempty(P.Data) P.Data = evalin('caller','R'); end
checkField(P,'Transpose',0);
checkField(P,'MirrorX',0);
checkField(P,'MirrorY',0);
checkField(P,'StepSize',0.1); % StepSize for moving with the buttons in seconds 
checkField(P,'Window',2); % A window of 2 seconds around
checkField(P,'FrameRate',[]);
checkField(P,'CMin',0);
checkField(P,'CMax',255);
checkField(P,'TrueLimits',0);
checkField(P,'Electrodes',1)
checkField(P,'StartFrame',1);
checkField(P,'CurvesByFrame',[]);
checkField(P,'Marker','x');
checkField(P,'SDAudio',NaN)
checkField(P,'FIG',1);
checkField(P,'ShowSensors',1);
checkField(P,'ShowAudio',1);
checkField(P,'ShowVideo',1);
checkField(P,'ShowEPhys',1);
checkField(P,'ShowGrid',1);
checkField(P,'ComputeLocation',0);
checkField(P,'DetectVocs',0);
checkField(P,'PhysicalSetup','GapCrossing');
checkField(P);

try delete(P.FIG); end
global MV; MV = []; 

switch P.PhysicalSetup
  case 'GapCrossing'; 
    P.CameraResolution = 0.00010938; % m/pixel
    P.CameraOffset = {[0,0,0],[256,320]}; % {XYZ Position, Pixel}
    P.MicrophonePositions = {[-0.23,0,0.314],[0.23,0,0.314]};
    P.CenterShift = 0;
  case {'FreeRunning1D','FreeRunning2D'};
    P.CameraResolution = 0.00065574; % m/pixel
    P.CameraOffset = {[0,0.03,0.25],[323,30]}; % {XYZ Position, Pixel}
    P.MirrorY = 1;
    P.Transpose = 1;
    P.CenterShift = 0;
    switch P.PhysicalSetup
      case 'FreeRunning1D'; P.MicrophonePositions = {[-0.296,0.279,0.383],[0.296,0.279,0.383]};
      case 'FreeRunning2D'; P.MicrophonePositions = {[-0.305,0.055,0.383],[0.305,0.055,0.383],[0,0.556,0.383]};
    end
  otherwise error('Option for PhysicalSetup not known.'); 
end

MV = transferFields(MV,P .Data); 
MV.Modules = setdiff( fieldnames(MV),'General');
MV.P = rmfield(P,'Data');
MV.FIG=P.FIG;
MV.Colors  = struct('NIDAQ',[0,0,0],'AnalogIn',[0,0,1],'PointGrey',[1,0,0],'ZeroLine',[1,0,0],'EPhys',[0,0,0],'LickP1',[1,0,0],'LickP2',[1,0,0]);
MV.VideoAvailable = any(strcmp(MV.Modules,'PointGrey')) * P.ShowVideo;
MV.AudioAvailable = any(strcmp(MV.Modules,'AnalogIn')) * P.ShowAudio;
MV.EPhysAvailable = any(strcmp(MV.Modules,'EPhys')) * P.ShowEPhys;
MV.VocPosition = NaN;

MV.Trials = MV.NIDAQ.Trials; MV.NTrials = length(MV.Trials); MV.NIDAQ.SRAI = 10000;
if any(diff(MV.Trials)>1) error('Only consecutive trials are supported in MultiViewer at thie point.'); end

for iT =1:MV.NTrials
  L = double(size(MV.NIDAQ.Data(iT).Data.Time,1));
  MV.DAQLengths(iT) = L/MV.NIDAQ.SRAI;
end
MV.DAQStarts = [0,cumsum(MV.DAQLengths(1:end-1))];
if MV.EPhysAvailable
  for iS=1:length(MV.NIDAQ.Data)
    MV.EPhysStarts(iS) = MV.EPhys.Data(iS).Data.Time(1);
  end
end
for iT=1:length(MV.NIDAQ.Data)
  if ~isempty(MV.NIDAQ.Data(iT).CameraStartTime)
    MV.VideoStarts(iT) = MV.NIDAQ.Data(iT).CameraStartTime(1);
  else 
    MV.VideoStarts(iT) = NaN;
  end
end
MV.TimeTotal = sum(MV.DAQLengths);
MV.StartTime = MV.NIDAQ.Data(1).Data.Time(1,1);
MV.StopTime = MV.NIDAQ.Data(end).Data.Time(end,1);

if P.MirrorX MV.PointGrey.Data.Data.Frames = flipdim(MV.PointGrey.Data.Data.Frames,2); end
if P.MirrorY MV.PointGrey.Data.Data.Frames = flipdim(MV.PointGrey.Data.Data.Frames,1); end
if P.Transpose MV.PointGrey.Data.Data.Frames = permute(MV.PointGrey.Data.Data.Frames,[2,1,3,4]); end

%% PREPARE FIGURE (COMPUTE SIZES)
SS = get(0,'ScreenSize');

figure(P.FIG); set(P.FIG,'Position',[50,400,SS(3:4)-[100,480]],'Toolbar','figure',...
  'Name',['MultiViewer ',MV.General.Parameters.General.Identifier],'NumberTitle','off',...
  'DeleteFcn',{@MV_saveData});
set(P.FIG,'KeyPressFcn',{@LF_KeyPress});
DC = axesDivide(1,[1,0.1],[0.05,0.04,0.93,0.9],[ ],[0.15]);
if MV.VideoAvailable
  DCTop = axesDivide([0.4,0.6],1,DC{1},[0.05],[]);
else
  DCTop{2} =  DC{1};
end
YDivision = []; YSep = []; LastInd = 0;
if MV.AudioAvailable 
  YDivision(end+[1:2]) = [0.3,0.3]; YSep(end+[1:2]) = [0.1,0.1]; 
  AudioInd = LastInd + [1:2]; LastInd = AudioInd(end);
end
if MV.EPhysAvailable 
  YDivision(end+[1]) = [0.2]; YSep(end+1) = [0.1];
  EPhysInd = LastInd+1; LastInd = EPhysInd; 
end
if MV.P.ShowSensors 
  YDivision(end+[1:2]) = [0.3,0.3]; YSep(end+[1:2]) = [0.1]; 
  SensorInd = LastInd + [1,2]; LastInd = SensorInd(end); 
end
DCTopRight = axesDivide(1,YDivision,DCTop{2},[],YSep);
DCBottom     = axesDivide([1.5,0.1,0.05,0.05,0.05,0.05],1,DC{2},[0.01],[]);

DS.Scrobble = DCBottom{1};
if MV.VideoAvailable  
  DS.Video                = DCTop{1}; 
  DS.Colorbar          = [DS.Video([1])+1.02*DS.Video([3]),DS.Video(2),0.01,DS.Video(4)];
end
if MV.AudioAvailable 
  DS.Spectrogram   = DCTopRight{AudioInd(1)}; 
  DS.Audio               = DCTopRight{(AudioInd(2))}; 
end
if MV.P.ShowSensors
  DS.Triggers           = DCTopRight{SensorInd(1)};
  DS.Sensors            = DCTopRight{SensorInd(2)};
end
if MV.EPhysAvailable
  DS.EPhys                = DCTopRight{EPhysInd};
end
colormap(gray(256)); 

F = fieldnames(DS);
for i=1:length(F); MV.AH.(F{i}) = axes('Pos',DS.(F{i})); box on; end

%% CONTROLS
MV.CurrentTime = 0;
MV.GUI.CurrentTime = uicontrol('style','edit','String',num2str(MV.CurrentTime),...
  'Units','normalized','Position',DCBottom{2},'Callback',{@MV_showData,'setedit'},'Tooltip','Current Time');
MV.GUI.StepSize = uicontrol('style','edit','String',num2str(MV.P.StepSize),...
  'Units','normalized','Position',DCBottom{3},'Tooltip','Step Size for Buttons');
MV.GUI.Help = uicontrol('style','pushbutton','String','Doc',...
  'Units','normalized','Position',DCBottom{4},'Tooltip','Show Documentation','Callback','doc MultiViewer');

ButtonStyles = {'pushbutton','pushbutton'};
Commands = {'Step','Step';'StepAnno','StepAnno'};
Strings = {'<','>'};
Tooltips = {'Step Forward','Step Backward'};
W = [1,1,1]; Colors = {0.8*W,0.8*W};
for i=1:length(Strings)
  UH(i) = uicontrol('style',ButtonStyles{i},'String',Strings{i},'Units','normalized','Position',DCBottom{4+i},...
    'Callback',{@MV_showData,Commands{1,i},Strings{i}},'BackGroundColor',Colors{i},'Tooltip',Tooltips{i},'KeypressFcn',@LF_KeyPress);
  uicontrol('style',ButtonStyles{i},'String',Strings{i},'Units','normalized','Position',DCBottom{4+i} + [0,0.08,0,-0.02],...
    'Callback',{@MV_showData,Commands{2,i},Strings{i}},'BackGroundColor',0.8*Colors{i},'Tooltip',Tooltips{i},'KeypressFcn',@LF_KeyPress);
end

%% EXPORT CONTROLS
MV.GUI.StartTime = uicontrol('style','edit','String',num2str(MV.StartTime),...
  'Units','normalized','Position',DCBottom{2}./[1,1,1,1.5]+[0,1.1*DCBottom{2}(4),0,0],'Tooltip','Export Start Time');
MV.GUI.StopTime = uicontrol('style','edit','String',num2str(MV.StartTime),...
  'Units','normalized','Position',DCBottom{2}./[1,1,1,1.5]+[1.1*DCBottom{2}(3),1.1*DCBottom{2}(4),0,0],'Tooltip','Export Stop Time');

if MV.P.ShowSensors
  %% PREPARE DATA DISPLAY
  InputNames = {MV.General.Paradigm.HW.Inputs.Name};
  axes(MV.AH.Sensors); set(MV.AH.Sensors,'ButtonDownFcn',{@MV_setTime}); hold on;
  MV.GUI.NIDAQ.SensorInd = find(~cellfun(@isempty,strfind(InputNames,'Pos')));
  MV.GUI.NIDAQ.SensorNames = InputNames(MV.GUI.NIDAQ.SensorInd);
  set(gca,'XLim',[-MV.P.Window/2,MV.P.Window/2],'YLim',[0,9]);
  plot([0,0],[-10,10],'Color',MV.Colors.ZeroLine);
  MV.Colors.AniPosP1S1 = [0.5,0,0];
  MV.Colors.AniPosP1S2 = [1,0,0];
  MV.Colors.AniPosP1S3 = [1,0.3,0.3];
  MV.Colors.PlatPosP1 = [1,0.6,0.6];
  MV.Colors.AniPosP2S1 = [0,0,0.5];
  MV.Colors.AniPosP2S2 = [0,0,1];
  MV.Colors.AniPosP2S3 = [0.3,0.3,1];
  MV.Colors.PlatPosP2 = [0.6,0.6,1];
  MV.Colors.CamPosC1 = [0.3,1,0.3];
  
  for i = 1:length(MV.GUI.NIDAQ.SensorInd)
    cColor = MV.Colors.(MV.GUI.NIDAQ.SensorNames{i});
    MV.GUI.NIDAQ.SensorH(i) = plot(0,0,'Color',cColor,'Hittest','off');
    text(0.02+(i-1)*0.11,0.9,MV.GUI.NIDAQ.SensorNames{i},'Units','n','FontSize',12,'FontWeight','bold','Color',cColor,'Hittest','off');
  end
  ylabel('Voltage [V]');
  xlabel('Time [s]');
  
  axes(MV.AH.Triggers); set(MV.AH.Triggers,'ButtonDownFcn',{@MV_setTime}); hold on;
  MV.GUI.NIDAQ.TriggerInd = setdiff([1:length(InputNames)],MV.GUI.NIDAQ.SensorInd);
  MV.GUI.NIDAQ.TriggerNames = InputNames(MV.GUI.NIDAQ.TriggerInd);
  MV.Colors.Trial = [1,0,0];
  MV.Colors.CamStart = [0,0,1];
  MV.Colors.CamTrigTo = [0,1,0];
  MV.Colors.CamTrigFrom = [0.5,1,0.5];
  MV.Colors.Feedback1 = [0,0,0];
  MV.Colors.Feedback2 = [0.5,0.5,0.5];
    
  set(gca,'XLim',[-MV.P.Window/2,MV.P.Window/2],'YLim',[-0.1,5.1]);
  plot([0,0],[-10,10],'Color',MV.Colors.ZeroLine);
  for i = 1:length(MV.GUI.NIDAQ.TriggerInd)
    cColor = MV.Colors.(MV.GUI.NIDAQ.TriggerNames{i});
    MV.GUI.NIDAQ.TriggerH(i) = plot(0,0,'Color',cColor,'Hittest','off');
    text(i*0.1,0.9,MV.GUI.NIDAQ.TriggerNames{i},'Units','n','FontSize',12,'FontWeight','bold','Color',cColor,'HitTest','off');
  end
  ylabel('Voltage [V]');
  MV_showNIDAQ;
end

if MV.AudioAvailable
  %% PREPARE AUDIO DISPLAY
  MV.AnalogIn.SRAI = MV.General.Parameters.Setup.Audio.SRAI;
  MV.AnalogIn.NChannels = size(MV.AnalogIn.Data(1).Data.Analog,2);
  SRAI = MV.AnalogIn.SRAI;
  axes(MV.AH.Audio); set(MV.AH.Audio,'ButtonDownFcn',{@MV_setTime}); hold on;
  if isnan(MV.P.SDAudio)
    for i=1: MV.AnalogIn.NChannels
      MV.GUI.AnalogIn.AudioStd(i) = std(MV.AnalogIn.Data(1).Data.Analog(1*SRAI:1.5*SRAI,i));
    end
  else
    MV.GUI.AnalogIn.AudioStd = repmat(MV.P.SDAudio,1,2);
  end
  
  NWindow = 0.05*SRAI;
  [b,a] = butter(2,[20000/(SRAI/2)],'high');
  for i=1:length(MV.AnalogIn.Data)
    if ~isempty(MV.AnalogIn.Data(i).Data.Analog)
      MV.AnalogIn.Data(i).Data.Analog = bsxfun(@times,MV.AnalogIn.Data(i).Data.Analog(:,1:MV.AnalogIn.NChannels),0.3./MV.GUI.AnalogIn.AudioStd);
    end
    NSteps = length(MV.AnalogIn.Data(i).Data.Analog);
    NSel = NWindow * floor(NSteps/NWindow);
    AnalogStd{i} = sum(abs(reshape(filter(b,a,MV.AnalogIn.Data(i).Data.Analog(1:NSel)),NWindow,NSel/NWindow)))/NWindow;
    TimeStd{i} = MV.AnalogIn.Data(i).Data.Time(NWindow/2:NWindow:NSel-NWindow/2);
    %   Ind =  find(MV.AnalogIn.Data(i).Data.Analog(:,1) > 2);
    %   DiffInd = Ind(find(diff(Ind) > 0.1*SRAI)+1);
    %   LoudPos{i} =  MV.AnalogIn.Data(i).Data.Time(DiffInd);
  end
  set(MV.AH.Audio,'YDir','normal','YLim',[-10,10],'XLim',[-MV.P.Window/2,MV.P.Window/2]);
  MV.Colors.AnalogInTrace = {[1,0,0],[0,0,1]}; MV.AnalogIn.Sides = {'L','R'};
  for i = 1:2
    MV.GUI.AnalogIn.SoundH(i) = plot(MV.AH.Audio,0,0,'-','Color',MV.Colors.AnalogInTrace{i},'Hittest','off');
    text(0.01,0.5+(i-1.5)/1.5160,MV.AnalogIn.Sides{i},'Units','n','FontSize',10,'FontWeight','bold','Color',MV.Colors.AnalogInTrace{i},'Hittest','off');
  end
  plot([0,0],[-10,10],'Color',MV.Colors.ZeroLine,'Hittest','off');
  ylabel('Voltage [V]');
  
  %% PREPARE SPECTROGRAM DISPLAY
  axes(MV.AH.Spectrogram); set(MV.AH.Spectrogram,'ButtonDownFcn',{@MV_setTime}); hold on;
  MV.GUI.AnalogIn.SpectrogramH = imagesc(0,0,0,'Hittest','off');
  plot([0,0],[0,MV.AnalogIn.SRAI/2],'Color',MV.Colors.ZeroLine,'HitTest','off');
  
  MV.AnalogIn.CorrTime = 0.0001;
  XTime = 1000*[-MV.AnalogIn.CorrTime:1/SRAI:MV.AnalogIn.CorrTime];
  MV.GUI.AnalogIn.XCorrH = plot(XTime,10000*ones(size(XTime)),'g','Hittest','off');
  MV.GUI.AnalogIn.DiffPosH = text(0.5,1.05,'','Color','r','FontSize',14,'FontWeight','bold','Horiz','center','HitTest','off','Units','n');
  set(MV.AH.Spectrogram,'YDir','normal','YLim',[0,125],'XLim',[-MV.P.Window/2,MV.P.Window/2]);
  ylabel('Freq. [kHz]');
  caxis([0,20]);
  MV.GUI.AnalogIn.SpecChannel = 1;
  MV.GUI.SpecChannel = text(0.02,0.06,MV.AnalogIn.Sides{MV.GUI.AnalogIn.SpecChannel},...
    'Units','n','FontSize',20,'FontWeight','Bold','Color',MV.Colors.AnalogInTrace{MV.GUI.AnalogIn.SpecChannel});
 
  %% ADD THE VOCALIZATIONS
  if MV.P.DetectVocs
    MV_DetectVocs; Opts = {'.','HitTest','off','MarkerSize',14};
    MV.GUI.AnalogIn.VocStarts = plot(zeros(size(MV.Vocs)),110*ones(size(MV.Vocs)),Opts{:},'Color','g');
    MV.GUI.AnalogIn.VocStops = plot(zeros(size(MV.Vocs)),110*ones(size(MV.Vocs)),Opts{:},'Color','r');
  end
  
  MV_showAudio;
end

%% PREPARE VIDEO DISPLAY
if MV.VideoAvailable
  axes(MV.AH.Video); hold on;
  % Dimensions are 512x640
  MV.PointGrey.Dims = size(MV.PointGrey.Data.Data.Frames); 
  MV.PointGrey.Dims = MV.PointGrey.Dims(1:2); 
  MV.GUI.Video.NFrames = 0;
  for i=1:length(MV.PointGrey.Data)
    MV.GUI.Video.NFrames = MV.GUI.Video.NFrames + MV.PointGrey.Data(i).Data.Dims(end);
  end
  if isempty(MV.P.CurvesByFrame) MV.CurvesByFrame = cell(MV.GUI.Video.NFrames,1); 
  else MV.CurvesByFrame = MV.P.CurvesByFrame;
  end
  MV.CurrentFrame = 0;
  
  % Dimensions are in meters
  MV.PointGrey.ViewDims = MV.P.CameraResolution * MV.PointGrey.Dims; 
  % Viewfield are the beginning and endpoint (Range X Dimensions)
  % Shifted to the Physical Coordinates
  MV.PointGrey.ViewField = [-MV.PointGrey.ViewDims/2;MV.PointGrey.ViewDims/2];
  NBinSteps  = 11;
  for i=1:2
    OffsetLocation(i) = MV.PointGrey.ViewField(1,i) + MV.P.CameraResolution*(MV.P.CameraOffset{2}(i)-1);
    MV.PointGrey.ViewField(1:2,i) = MV.PointGrey.ViewField(1:2,i) - OffsetLocation(i) + MV.P.CameraOffset{1}(i);
    Steps{i} = linspace(MV.PointGrey.ViewField(1,i),MV.PointGrey.ViewField(2,i),MV.PointGrey.Dims(i));
    Ticks{i} = HF_autobin(MV.PointGrey.ViewField(1,i),MV.PointGrey.ViewField(2,i),NBinSteps);
  end
  MV.GUI.Video.FrameH = imagesc(Steps{1},Steps{2},zeros(MV.PointGrey.Dims));
  set(MV.AH.Video,'YDir','normal');
  set(MV.GUI.Video.FrameH,'HitTest','off');  hold on;
  
  set(MV.AH.Video,'XTick',Ticks{1},'YTick',Ticks{2},...
    'PlotBoxAspectRatio',[MV.PointGrey.Dims./max(MV.PointGrey.Dims),1],...
    'PlotBoxAspectRatioMode','manual',...
    'ButtonDownFcn',{@MV_mainFrame},'XLim',MV.PointGrey.ViewField(:,1)+[-0.01,0.01]','YLim',MV.PointGrey.ViewField(:,2)+[-0.01,0.01]');
  
  if MV.P.ShowGrid
    for i=1:length(Ticks{1})
      plot3([Ticks{1}(i),Ticks{1}(i)],Ticks{2}([1,end]),[256,256],':','Color','w','HitTest','off');
    end
  end
  MV.GUI.Video.VocLocation = plot3([-40,-40],Ticks{2}([1,end]),[256,256],'-','Color','g','Hittest','off');
  xlabel('X [m]'); ylabel('Y [m]');
%  axis([MV.PointGrey.ViewField(1:2,1)',MV.PointGrey.ViewField(1:2,2)']);
  caxis([100,255]);
  MV.GUI.Video.TitleH = title(['Video']);
  MV.GUI.AllAnnotations = plot([-1000],[-1000],'.r'); 
  set(MV.GUI.AllAnnotations,'HitTest','off');
  MV.CurrentInspection = 0;
  
  %% COLORBAR & COLOR CONTROLS
  axes(MV.AH.Colorbar);
  MV.NColSteps = 256; colormap(gray(MV.NColSteps)); MV.Colormap =0;
  MV.CLimits = [MV.P.CMin,MV.P.CMax]; 
  MV.GUI.CIH = imagesc(1); set(MV.GUI.CIH,'Hittest','off');
  set(MV.AH.Colorbar,'ButtonDownFcn',{@MV_setCLimit,'setvisual'})
  MV_setCLimit;
end

%% PREPARE EPHYS DISPLAY
if MV.EPhysAvailable
  SRAI = MV.EPhys.SRAI;
  NElectrodes = length(MV.P.Electrodes);
  Range = 200;
  axes(MV.AH.EPhys); set(MV.AH.EPhys,'ButtonDownFcn',{@MV_setTime}); hold on;
  set(MV.AH.EPhys,'YDir','normal','YLim',[0,Range*NElectrodes],'XLim',[-MV.P.Window/2,MV.P.Window/2],'YTick',[]);
  
  for iT = 1:length(MV.EPhys.Data) 
    MV.EPhys.Data(iT).Data.Analog = MV.EPhys.Data(iT).Data.Spike;
    MV.EPhys.Data(iT).Data = rmfield(MV.EPhys.Data(iT).Data,'Spike');
  end
  
  for iE=1:NElectrodes
    if mod(floor((iE-1)/4),2)==0 cColor = [0,0,0.5]; else cColor = [0.5,0,0]; end
    MV.GUI.EPhys.DataH(iE) = plot(MV.AH.EPhys,0,0,'-','Color',cColor,'Hittest','off');
    text(-1.01,Range*(iE-0.5),['El.',num2str(MV.P.Electrodes(iE))],...
      'FontSize',12,'FontWeight','bold','Color',cColor,'Hittest','off','Horiz','right');
  end
  plot([0,0],[0,(iE+1)*Range],'r');
  xlabel('Time [s]');
  
  MV.GUI.EPhys.Electrodes = MV.P.Electrodes;
  ylabel('Voltage [\muV]','Units','n','Position',[-0.05,0.5]);
end

%% CREATE SCROBBLER
% Plot Markers at the Trial ends (red bars)
axes(MV.AH.Scrobble); hold on;
plot3([MV.CurrentTime,MV.CurrentTime],[0,1],[1,1],'r','Hittest','off'); 
axis([MV.StartTime,MV.StopTime,0,1]);
% SHOW ANIMAL POSITION
if ~isempty(MV.General.Paradigm.HW.AnimalSensorsPhys)
  [AnimalPositionVsTime , AnimalTime , AnimalPositions ] = MV_computeAnimalPosition;
  imagesc(AnimalTime,AnimalPositions,AnimalPositionVsTime,'HitTest','off');
  caxis([-1,1]);
end

for iT = 1:MV.NTrials
  if ~isempty(MV.NIDAQ.Data(iT).TrialStartTime)
    MV.TrialStarts(iT) = double(MV.NIDAQ.Data(iT).TrialStartTime);
    MV.TrialStops(iT) = double(MV.NIDAQ.Data(iT).TrialStopTime);
    patch([MV.TrialStarts(iT),MV.TrialStops(iT),MV.TrialStops(iT),MV.TrialStarts(iT)],[0,0,1,1],[0.1,0.1,0.1,0.1],0,'FaceColor',[0.8,0.8,0.8],'HitTest','off','facealpha',0.5)
    text(MV.TrialStarts(iT),1.3,num2str(iT),'Color','k','Horiz','center','FontSize',14);
  end
end

% Plot Scrobbler bar
MV.GUI.CurrentLine = plot3(repmat(MV.CurrentTime,1,2), [0,1],[1,1],'Color',MV.Colors.ZeroLine,'HitTest','off','LineWidth',2);
if MV.VideoAvailable set(MV.AH.Video,'ZLim',[0,256]); end

% Plot horizontal bars in different colors to show when different modalities were recorded
for iT=1:MV.NTrials
  for iM=1:length(MV.Modules)
    cModule = MV.Modules{iM};
    cColor = MV.Colors.(cModule);
    ModPos = iM/(length(MV.Modules)+1);
    if iT == 1
      text(-1,ModPos,iM,[cModule,' '],'Horiz','right','Color',cColor,'FontWeight','bold');
    end
    if length(MV.(cModule).Data)>=iT && ...
        ~isempty(MV.(cModule).Data(iT).Data.Time)
        cTimes = MV.(cModule).Data(iT).Data.Time([1,end],1);
          plot(cTimes,repmat(ModPos,1,2),'.-','Color',cColor,'LineWidth',1.5,'HitTest','off');
      switch cModule
        case 'PointGrey'; % CHECK FOR INTERNAL STOPS WHERE NO VIDEO WAS ACQUIRED (TO BE DONE)
          InterTimes = diff(MV.(cModule).Data(iT).Data.Time(:,1));
          cDiffInds= [0;find(InterTimes>0.1);length(InterTimes)+1];
          for iP = 1:length(cDiffInds)-1
            cInds = [cDiffInds(iP)+1,cDiffInds(iP+1)];
            cTimes = MV.(cModule).Data(iT).Data.Time(cInds,1);
            plot(cTimes,repmat(ModPos,1,2),'.-','Color',0.5*cColor,'LineWidth',1.5,'HitTest','off','MarkerSize',16);
          end
        case 'AnalogIn';
          if MV.AudioAvailable
            for i=1:length(TimeStd)
              cInd = intersect(find(AnalogStd{i}>0.25),find(AnalogStd{i}<1));
              plot(TimeStd{i}(cInd),repmat(ModPos,size(cInd)),'.','Color',cColor,'MarkerSize',16,'HitTest','off')
            end
          end
      end
    end
  end
end


% Activate Scrobbler to choose point in time
text(0,-0.25,'Time [s]  ','Units','n','FontWeight','bold','Horiz','right');
XTicks = C_autobin(MV.StartTime,MV.StopTime,10);
set(MV.AH.Scrobble,'ButtonDownFcn',{@MV_scrobblor},'XTick',XTicks,'YTick',[],'XGrid','on');
set(P.FIG,'WindowButtonUpFcn','global Scrobbling_ ; Scrobbling_ = 0;');

%% CALLBACK FUNCTIONS
function MV_mainFrame(O,E)
global MV;

SelType = get(MV.FIG,'SelectionType');
switch SelType
  case 'normal'; % Mark left animal
    MV_recordLocation(O,E,'left');
  case 'alt'; 
    MV_recordLocation(O,E,'right');
  case 'open';
    if isfield(MV,'CurrentFrame')
      MV.CurvesByFrame{MV.CurrentFrame}{end}{4} = 'whisker';
      MV_recordLocation([],[],'refresh');
      MV_showVideo;
    end
  case 'extend'
    MV_zoomFrame(O,E);
end

function MV_zoomFrame(O,E)
global MV;
Point = get(O,'CurrentPoint');
Point = Point(1,1:2);
Dims = MV.PointGrey.Dims;
set(MV.AH.Video,...
  'XLim',[Point(1) - Dims(2)/4,Point(1)+Dims(2)/4],...
  'YLim',[Point(2) - Dims(1)/4,Point(2)+Dims(1)/4]);

function MV_recordLocation(O,E,Opt)
global MV;
if ~exist('Opt','var') Opt = ''; end
switch Opt
  case 'refresh';
  otherwise
    Point = get(O,'CurrentPoint');
    Point = Point(1,1:2);
    switch architecture
      case 'MAC';
        % LOCATION IS  CONSISTENTLY IMPRECISE, EMPIRICAL CORRECTION
        Point = Point + [ -0.2/56 , 0.48/70 ] .* MV.PointGrey.ViewDims;
    end
    fprintf(['X : ',num2str(Point(1)),'    Y : ',num2str(Point(2)),'\n']);
    if isfield(MV,'CurrentFrame')
      MV.CurvesByFrame{MV.CurrentFrame}{end+1} = ...
        {Point(1),Point(2),now,Opt,MV.CurrentTime,MV.VocPosition};
    end
end

XData = find(~cellfun(@isempty,MV.CurvesByFrame));
set(MV.GUI.AllAnnotations,'XData',XData,'YData',zeros(size(XData)) + 1);
MV_showVideo;

%% CALLBACK FUNCTIONS
function LF_KeyPress(handle,event,FID)

global MV
CC = get(MV.FIG,'CurrentCharacter');
if ~isempty(CC)
  switch int8(CC)
    case 28; MV_showData([],[],'step','<');  % left
    case 29; MV_showData([],[],'step','>'); % right
    case 100; %d Delete Annotations for current Frame
      MV.CurvesByFrame{MV.CurrentFrame} = {};
      MV.CurrentInspection = MV.CurrentInspection -1;
      MV_recordLocation([],[],'refresh');
      MV_showVideo;
    case 117; %u : undo last Annotation
      MV.CurvesByFrame{MV.CurrentFrame} = MV.CurvesByFrame{MV.CurrentFrame}(1:end-1);
      MV_recordLocation([],[],'refresh');
      MV_showVideo;
    case 113; set(MV.GUI.StartFrame,'String',num2str(MV.CurrentTime)); MV_setLimit(MV.GUI.StartFrame,[],'start');
    case 119; set(MV.GUI.StopFrame,'String',num2str(MV.CurrentTime)); MV_setLimit(MV.GUI.StopFrame,[],'end');
    case 110; MV_exportMovie([],[],'getframe'); %n export movie using getframe
    case 83; 
      if MV.GUI.AnalogIn.SpecChannel == 1  MV.GUI.AnalogIn.SpecChannel=2; else  MV.GUI.AnalogIn.SpecChannel=1; end
      MV_showData([],[],'redraw',''); 
    case 115; MV_saveData; % s save annotations before quitting  
    case 32;  % space = play
      switch get(MV.GUI.PlayForward,'Value')
        case 0; set(MV.GUI.PlayForward,'Value',1); MV_showData([],[],'play','>');
        case 1; set(MV.GUI.PlayForward,'Value',0);
      end
    case {43,61}; % =/+
      CLim = get(MV.AH.Spectrogram,'CLim'); set(MV.AH.Spectrogram,'CLim',1.2*CLim);
    case {45,95}; % -/_
      CLim = get(MV.AH.Spectrogram,'CLim'); set(MV.AH.Spectrogram,'CLim',0.8*CLim);
  end
end
  
function MV_showData(O,E,Command,iFrame)
global MV

switch lower(Command)
  case 'set'; MV.CurrentTime = iFrame;
  case 'setedit'; MV.CurrentTime= str2num(get(O,'String'));
  case 'setvisual'; MV.CurrentTime = get(O,'CurrentPoint');MV.CurrentTime = round(MV.CurrentTime(1,1));
  case 'step';   
    MV.P.StepSize = str2num(get(MV.GUI.StepSize,'String'));
    switch iFrame
      case '>'; MV.CurrentTime = MV.CurrentTime + MV.P.StepSize; 
      case '<'; MV.CurrentTime = MV.CurrentTime - MV.P.StepSize; 
    end
  case 'stepanno'; % Step to the next annotation
    if MV.VideoAvailable
      AnnotatedFrames = find(~cellfun(@isempty,MV.CurvesByFrame));
      FrameDist = AnnotatedFrames - MV.CurrentFrame;
      switch iFrame
        case '>';  FrameDist(FrameDist <= 0) = inf; [~,Pos] = min(FrameDist);
        case '<';  FrameDist(FrameDist >= 0) = -inf; [~,Pos] = max(FrameDist);
      end
      NewFrame = AnnotatedFrames(Pos);
      MV.CurrentTime = MV.PointGrey.Data.Data.Time(NewFrame);
    end
  case 'redraw';
end

MV.CurrentTime = min([MV.StopTime,MV.CurrentTime]);
MV.CurrentTime= max([MV.StartTime,MV.CurrentTime]);

set(MV.GUI.CurrentTime,'String',num2str(MV.CurrentTime));
set(MV.GUI.CurrentLine,'XData',[MV.CurrentTime,MV.CurrentTime]);

if MV.P.ShowSensors MV_showNIDAQ; end % Shows NIDAQ ( SENSORS and TRIGGERS )
if MV.AudioAvailable MV_showAudio; end % Shows AnalogIn (SoundPressure and Spectrogram) 
if MV.VideoAvailable MV_showVideo; end % Shows Video Data (FRAME)
if MV.EPhysAvailable MV_showEPhys; end % Shows EPhys Data (FRAME)

function MV_exportMovie(O,E,Opt)
 global MV;
 
  switch Opt
    case {'getframe'};
      StartTime =  str2num(get(MV.GUI.StartTime,'String'));
      StopTime =  str2num(get(MV.GUI.StopTime,'String'));
      StepSize =  str2num(get(MV.GUI.StepSize,'String'));

    otherwise
  end
 
  switch Opt
    case 'getframe';
      Filename = input('Specify Filename for Movie: ','s');
      switch architecture
        case 'PCWIN'; 
          [R,S]  =system('echo %HOMEPATH%');
          Dir = ['C:',S(1:end-1),filesep,'Desktop',filesep];
        case 'MAC';
          Dir = ['~/Desktop/'];
      end
      V = VideoWriter([Dir,Filename,'.avi']);
      V.open; CR = [];
      FigureSize = get(MV.FIG,'Position');
      Rectangle = [0,170,FigureSize(3),FigureSize(4)-170];
      for cTime=StartTime:StepSize:StopTime
        R= fprintf([CR,'Time ',num2str(cTime)]);
        R = R-length(CR)/2;
        MV_showData([],[],'set',cTime);
        F = getframe(MV.FIG,Rectangle);
        V.writeVideo(F);
        CR = repmat('\b',1,R);
      end
      V.close; fprintf('\n');
  end

function [cTime,cData] = MV_getCurrentData(Module)
  global MV
  StartTime = MV.CurrentTime-MV.P.Window/2;
  StopTime = MV.CurrentTime+MV.P.Window/2;
  switch Module
    case {'NIDAQ','AnalogIn','Video'}; Starts = MV.DAQStarts;
    case 'EPhys'; Starts = MV.EPhysStarts;
  end
 
  TrialStart = find(Starts<StartTime,1,'last');
  TrialStop = find(StopTime>Starts,1,'last');
  FillZerosStart = isempty(TrialStart);
  FillZerosStop = isempty(TrialStop);
   
   % FIND THE STARTING AND STOPPING POINT FOR THE PLOTTING (USES COARSE
   % SEARCH FIRST FOR SPEED UP);
   Factor = 1000;
   if FillZerosStart;     cIndStart = [];
   else
     cIndStartCoarse = find(MV.(Module).Data(TrialStart).Data.Time(Factor:Factor:end,1) >= StartTime,1,'first');
     if ~isempty(cIndStartCoarse)
       cInds = cIndStartCoarse*Factor+[-Factor+1:0];
       cIndStartFine = find(MV.(Module).Data(TrialStart).Data.Time(cInds,1) >= StartTime,1,'first');
       cIndStart = (cIndStartCoarse-1)*Factor + cIndStartFine;
     else 
       cIndStart = [];
     end
   %    IndStart = find_halfspace_mex(MV.(Module).Times,StartTime);
   end
   
   if FillZerosStop;      cIndStop = [];
   else
     cIndStopCoarse = find(MV.(Module).Data(TrialStop).Data.Time(Factor:Factor:end,1) <= StopTime,1,'last');
     if ~isempty(cIndStopCoarse)
       cInds = [(cIndStopCoarse-1)*Factor+1  : min((cIndStopCoarse+1)*Factor,length(MV.(Module).Data(TrialStop).Data.Time))];
       cIndStopFine = find(MV.(Module).Data(TrialStop).Data.Time(cInds,1) <= StopTime,1,'last');
       cIndStop = (cIndStopCoarse-1)*Factor + cIndStopFine;
     else 
       cIndStop = [];
     end
    % cIndStop = find(MV.(Module).Data(TrialStop).Data.Time(:,1) <= StopTime,1,'last');
   end

   if TrialStart == TrialStop
     Inds = [cIndStart:cIndStop];
     cTime = MV.(Module).Data(TrialStart).Data.Time(Inds,1);
     cData = MV.(Module).Data(TrialStart).Data.Analog(Inds,:);
   else
     if FillZerosStart
       cTime = MV.(Module).Data(TrialStop).Data.Time(1:cIndStop,1);
       cData = MV.(Module).Data(TrialStop).Data.Analog(1:cIndStop,:);
     elseif FillZerosStop
       cTime = MV.(Module).Data(TrialStart).Data.Time(cIndStart:end,1);
       cData = MV.(Module).Data(TrialStart).Data.Analog(cIndStart:end,:);       
     else
       cTime = ...
         [MV.(Module).Data(TrialStart).Data.Time(cIndStart:end,1);...
         MV.(Module).Data(TrialStop).Data.Time(1:cIndStop,1)];
       cData = ...
         [MV.(Module).Data(TrialStart).Data.Analog(cIndStart:end,:);...
         MV.(Module).Data(TrialStop).Data.Analog(1:cIndStop,:)];
     end
   end
   % find_halfspace_mex finds the next index below the closest index (from below)
   % cIndStart = find(MV.(Module).Times > MV.CurrentTime-MV.P.Window/2,1,'first');
   %   IndStart = find_halfspace_mex(MV.(Module).Times,MV.CurrentTime-MV.P.Window/2);
   % cIndStop = find(MV.(Module).Times < MV.CurrentTime+MV.P.Window/2,1,'last');
   %   IndStop = find_halfspace_mex(MV.(Module).Times,MV.CurrentTime+MV.P.Window/2);
   if size(cTime,1) ~= size(cData,1) keyboard; end   
  
function MV_showNIDAQ
   global MV;
 
   [cTime,cData] = MV_getCurrentData('NIDAQ');
   
   % FILL THE SENSOR PLOT
   for i=1:length(MV.GUI.NIDAQ.SensorInd)
     set(MV.GUI.NIDAQ.SensorH(i),'YData',cData(:,MV.GUI.NIDAQ.SensorInd(i)) +3*floor((i-1)/4),'XData',cTime-MV.CurrentTime);
   end

   % FILL THE TRIGGER PLOT
   for i=1:length(MV.GUI.NIDAQ.TriggerInd)
     set(MV.GUI.NIDAQ.TriggerH(i),'YData',cData(1:2:end,MV.GUI.NIDAQ.TriggerInd(i))/5+1.1*(i-1),'XData',cTime(1:2:end)-MV.CurrentTime);
   end

   
function MV_showEPhys
  global MV;
  
  [cTime,cData] = MV_getCurrentData('EPhys');
  
  % FILL THE ELECTRODE PLOT
  for iE=1:length(MV.P.Electrodes)
    set(MV.GUI.EPhys.DataH(iE),'YData',cData(:,iE)+(iE-0.5)*200,'XData',cTime-MV.CurrentTime);
  end

   
function MV_showAudio
  global MV;
  SRAI = MV.General.Parameters.Setup.Audio.SRAI;
  NFFT = 512;
  
  [cTime,cData] = MV_getCurrentData('AnalogIn');

  if ~isempty(cTime)
    % FILL THE TRIGGER PLOT
    if MV.AnalogIn.NChannels > 1 ShiftRange = 3; else ShiftRange = 1; end
    for i=1:MV.AnalogIn.NChannels
      Shift = i*ShiftRange/MV.AnalogIn.NChannels - ShiftRange/2;
      set(MV.GUI.AnalogIn.SoundH(i),'YData',cData(1:1:end,i) + Shift,'XData',cTime(1:1:end)-MV.CurrentTime);
    end
    if length(cData) > NFFT
     [ S , F ,T] = spectrogram(double(cData(:,MV.GUI.AnalogIn.SpecChannel)),NFFT,NFFT/2,NFFT,SRAI);
     set(MV.GUI.AnalogIn.SpectrogramH,'CData',abs(S),'XData',T- (MV.CurrentTime-cTime(1)),'YData',F/1000);
   end
   % fprintf('NIDAQ: %f  Audio : %f  Video: %f \n',T(1),T(2),T(3));
  else 
    for i=1:MV.AnalogIn.NChannels
      set(MV.GUI.AnalogIn.SoundH(i),'YData',[],'XData',[]);
    end
    set(MV.GUI.AnalogIn.SpectrogramH,'CData',NaN);
  end
  
  set(MV.GUI.SpecChannel,'String',MV.AnalogIn.Sides{MV.GUI.AnalogIn.SpecChannel},...
    'Color',(MV.Colors.AnalogInTrace{MV.GUI.AnalogIn.SpecChannel}+1)/2);

  % PLOT VOCALIZATION
  if MV.P.DetectVocs && numel(MV.Vocs) > 0
    Starts = [MV.Vocs.Start] - MV.CurrentTime;
    Stops = [MV.Vocs.Stop] - MV.CurrentTime;
    set(MV.GUI.AnalogIn.VocStarts,'XData',Starts);
    set(MV.GUI.AnalogIn.VocStops,'XData',Stops);
  end
  
  % PLOT XCORR AUDIO
  if MV.P.ComputeLocation
    if ~isempty(cTime)
      ccTime = cTime - MV.CurrentTime;
      Ind = abs(ccTime) < 0.03;
      ccData = cData(Ind,:);
      if length(ccData) > 10
        LocMethod = 'Spectral';
        [StimPosEst,MidDist,SCorr,Time,DeltaTime] = VocLocalizer(...
          'Sounds',{ccData(:,1),ccData(:,2)},'HighPass',25000,...
          'SR',SRAI,'CorrTime',2*MV.AnalogIn.CorrTime,'CorrMethod',{'GCC','XCorr'},...
          'DelayRange',2*MV.AnalogIn.CorrTime,...
          'MicrophonePositions',MV.P.MicrophonePositions,...
          'SourceHeight',MV.P.CameraOffset{1}(3),...
          'CenterShift',MV.P.CenterShift,'FIG',0);
        MV.VocPosition = StimPosEst;
        SCorr = 125*SCorr/(3*max(SCorr));
        set(MV.GUI.AnalogIn.XCorrH,'YData',SCorr,'XData',1000*Time);
       set(MV.GUI.AnalogIn.DiffPosH,'String',[sprintf('%0.4f',StimPosEst(1)),'m']);
       if MV.VideoAvailable set(MV.GUI.Video.VocLocation,'XData',repmat(StimPosEst(1),1,2)); end
        %disp(['Distance to Center: ',num2str(DeltaTime*1000),'ms = ',num2str(MidDist*1000),'mm']);
      end
    end
  end
  
function MV_showVideo
  global MV;
  
  Trial = find(MV.CurrentTime>MV.VideoStarts,1,'last');
  if ~isempty(Trial) && Trial > 0
    cFrame = find(MV.PointGrey.Data(Trial).Data.Time(:,1) >= MV.CurrentTime,1,'first');
    cTime = MV.PointGrey.Data(Trial).Data.Time(cFrame,1);
    MV.CurrentFrame = cFrame;
    if ~isempty(cFrame) && abs(cTime - MV.CurrentTime) <0.05 % if within 20ms of an existing frame
      cData = MV.PointGrey.Data(Trial).Data.Frames(:,:,1,cFrame);
    else
      cData = zeros(MV.PointGrey.Dims);
      cFrame = NaN;
    end
    
    % OUTPUT FRAME TO FRAME DIFFERENCE IN BRIGHTNESS
    R = sort(cData(:));
     CorrFact = mean(R(end-round(0.01*length(R)):end));
    CorrFact = CorrFact/220;
     cData = double(cData)/CorrFact;

    %% SHOW OTHER INFORMATION (WHISKER FITS / CONTACTS)
    try delete(MV.LastPH); end
    MV.LastPH = [];
    if ~isempty(MV.CurrentFrame)
      cCurves = MV.CurvesByFrame{MV.CurrentFrame};
      if ~isempty(cCurves)
        for i=1:length(cCurves)
          switch cCurves{i}{4}
            case 'left'; cColor = [0.3,0.6,0.3];
            case 'right'; cColor = [0.9,0.3,0.2];
            case 'whisker'; cColor = [1,1,1];
            otherwise cColor = [1,1,1];
          end
          MV.LastPH(end+1) = plot(MV.AH.Video,...
            cCurves{i}{1},cCurves{i}{2},MV.P.Marker,...
            'Color',cColor,'MarkerSize',8,'LineWidth',2,'Hittest','off');
        end
      end
    end
    MV.CurrentInspection = MV.CurrentInspection + 1;
    MV.SequenceOfInspection(MV.CurrentInspection,[1:2]) = [now,MV.CurrentFrame]; 
  else
      cData = zeros(MV.PointGrey.Dims);
      cFrame = NaN;
  end

  set(MV.GUI.Video.FrameH,'CData',cData');
  set(MV.GUI.Video.TitleH,'String',['Video : Trial ',num2str(Trial),' Frame : ' num2str(cFrame),' |'])

  
function MV_saveData(O,E)
  global MV
  if isempty(MV) || ~isfield(MV,'SequenceOfInspection') return; end
  R.CurvesByFrame = MV.CurvesByFrame;
  R.SequenceOfInspection = MV.SequenceOfInspection(1:MV.CurrentInspection,:);
  AnnotationsExist = sum(~cellfun(@isempty,R.CurvesByFrame));
  
  if AnnotationsExist
    if ~isfield(MV,'FileNameAnnotations')
      MV.FileNameAnnotations = input('Enter Filename for Annotations : ','s');
      if isempty(MV.FileNameAnnotations) return; end
      MV.FileNameAnnotations = [pwd,filesep,MV.FileNameAnnotations];
    end
    if isfield(MV,'FileNameAnnotations')
      disp(['Saving to ',MV.FileNameAnnotations]);
      save(MV.FileNameAnnotations,'-struct','R');
    end
  end
  
function MV_scrobblor(O,E)

  global MV;
  
  AxPos = get(MV.AH.Scrobble,'Position');
  FigPos = get(MV.FIG,'Position');
  Pixels = AxPos(3)*FigPos(3);
  TimeTotal = MV.StopTime-MV.StartTime;
  TimePerPixel = TimeTotal/Pixels;
  AxisStartPixels =  FigPos(1) + AxPos(1)*FigPos(3);
 % LastPoint = [-inf,-inf];
  
  SelType = get(MV.FIG,'SelectionType');
  switch SelType
    case 'normal';
      global Scrobbling_ ; Scrobbling_ =1;
      
      while Scrobbling_
        CurrentPoint = get(0,'PointerLocation');
        CurrentTime  = TimePerPixel*(CurrentPoint(1)-AxisStartPixels) + MV.StartTime;
        MV_showData([],[],'set',CurrentTime);
        global Scrobbling_
        pause(0.1);
      end
      
    case {'alt','extend'};
      global Scrobbling_ ; Scrobbling_ =0;
  end
  
  function MV_setTime(O,E)
  global MV;
  
  Position = get(gca,'CurrentPoint');
  NewTime = MV.CurrentTime + Position(1);
  
  SelType = get(MV.FIG,'SelectionType');
  switch SelType
    case {'alt','extend'};
      MV.P.ComputeLocation = 0;
  end;
  
  MV_showData([],[],'set',NewTime);
  if MV.AnalogIn.NChannels > 1; MV.P.ComputeLocation = 1; end

  function [AnimalPositionsVsTime,AnimalTime,AnimalPositions] = MV_computeAnimalPosition
    global MV;
    Pos = C_estimateAnimalPosition('Data',MV,'BetweenSensors',0,'SR',10);
    AnimalPositionsVsTime = Pos.AnimalPositions;
    AnimalTime = Pos.Time;
    AnimalPositions = Pos.Sensors;
    NPos =size(AnimalPositionsVsTime,1);
    AnimalPositions = linspace(1/(2*NPos), 1-1/(2*NPos),NPos);    
    
function MV_setCLimit(O,E,Opt,CLimits)
global MV;
if ~exist('Opt','var') Opt = ''; end

switch Opt
  case 'set'; MV.CLimits = CLimits;
  case 'setedit'; 
    if O==MV.GUI.CLimMin LimInd = 1; else LimInd = 2; end 
    MV.CLimits(LimInd) = str2num(get(O,'String'));
  case 'setvisual';
    NewLimit = get(MV.AH.Colorbar,'CurrentPoint');
    NewLimit = round(255*NewLimit(1,2));
    SelType = get(MV.FIG,'SelectionType');
    switch SelType
      case 'normal'; MV.CLimits(1) = NewLimit;
      case 'alt'; MV.CLimits(2) = NewLimit;
    end
  otherwise % for empty case
end

MV.CLimits = sort(MV.CLimits);
CY = linspace(MV.P.CMin,MV.P.CMax,MV.NColSteps);
NBelow =  round(MV.NColSteps*(MV.CLimits(1)-MV.P.CMin)/(MV.P.CMax-MV.P.CMin));
NAbove = round(MV.NColSteps*(MV.P.CMax-MV.CLimits(2))/(MV.P.CMax-MV.P.CMin));
CY = [zeros(1,NBelow) , linspace(0,1,MV.NColSteps-NBelow-NAbove) , ones(1,NAbove)]' ;
set(MV.GUI.CIH,'CData',CY,'YData',CY,'XData',1); caxis(MV.AH.Colorbar,[0,1]);
set(MV.AH.Colorbar,'YLim',CY([1,end]),'YTick',[]);
caxis(MV.AH.Video,MV.CLimits);


  function MV_DetectVocs
    global MV;
    if ~isfield(MV.AnalogIn,'FromWAV')
      cAnimal = MV.General.Parameters.General.Animal;
      cRecording = MV.General.Parameters.General.Recording;
      MV.Vocs = VocCollector('Animals',{cAnimal},'Recording',cRecording);
    else
      MV.Vocs = VocCollector('DataSource','Direct','Data',MV.AnalogIn.Data.Data.Analog,'SRAI',MV.AnalogIn.SRAI);
    end