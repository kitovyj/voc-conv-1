function FIG_USVClassify_GenderNetwork(varargin)

%% PARSE ARGUMENTS
P = parsePairs(varargin);
checkField(P,'FIG',1); 
checkField(P,'Save',1);
checkField(P,'View',0);
checkField(P,'Recompute',0);  

%% SETUP BASICS
Opt = FIG_USVLoc;
setPlotOpt(Opt.Journal,'path',Opt.Dir,'cols',2,'height',9); 

if P.Recompute  LF_recomputeData(P.Recompute,inpath,name); end; 
if exist([inpath,name,'.mat'],'file')  R = load([inpath,name]); end

%% PREPARE FIGURE
figure(P.FIG); clf; set(P.FIG,FigOpt{:}); HF_matchAspectRatio;
DCBase = axesDivide([1],[1.3,1],[0.08,0.1,0.9,0.85],[ ],[0.7]);
DC1 = axesDivide([1,0.8,1],[1],DCBase{1},[0.5,0.5],[]); 
DC2 = axesDivide([3],[1],DCBase{2},[0.6],[])';
DC = [DC1(:);DC2(:)];

Labels = {'A','B','C','D','E','F'}; LdPos = [-0.05,0.025];
for iS = 1:numel(DC)
  AH(iS) = axes('Pos',DC{iS}); hold on; FigLabel(Labels{iS},LdPos); 
  box off;
end
HF_setFigProps;

% START PLOTTING 
Titles = {'Male','Female','Amplitude','Duration','Frequency','Spectral Range'};
P.SelectByEnergyDiff = 1;
MarkerSize = 6;
LineWidth = 2;

% Random controls are always grayed out by 50%
Colors = struct('DNNFull',[1,0,1],'Chance',[0.5,0.5,0.5],'LR',[0,1,0],'SVM',[0.4,1,0],...
  'Male',[0,0,1],'Female',[1,0,0],'Random',[0.35,0.35,0.35]);

iA = 1;

% ARCHITECTURE... probably not plot everything, maybe some parts, and the label
axes(AH(iA)); iA = iA + 1;

% TRAINING CONVERGENCE
axes(AH(iA)); iA = iA + 1;
cX = [1:1000];
cY = [cX.^0.3]; cY = 100*cY/cY(end);
plot(cX([1,end]),[10,10],'LineWidth',LineWidth,'Color',Colors.Chance); % Line indicating Chance Performance
plot(cX,cY,'LineWidth',LineWidth,'Color',Colors.DNNFull); % Training success for the full network
axis tight; ylim([0,100]);
xlabel('Iterations'); ylabel('Performance [%]');

% SAMPLE RECEPTIVE FIELDS
% maybe needs to be subdivided
axes(AH(iA)); iA = iA + 1;
imagesc(rand(10));
axis tight;
colorbar;

% SHOW ESTIMATION RESULTS
Plots = {'Gender Network','Linear Prediction',''};
DataByPlot = {{'DNNFull','Female','Male','Random'},{'LR','Random','SVM','Random'},{'DNNFull','DNNFull','DNNFull','DNNFull'}};
for iP=1:length(Plots); % LOOP OVER NETWORK TYPES
  axes(AH(iA)); iA = iA + 1;

  for iD = 1:length(DataByPlot{iP}) % LOOP OVER DATA INSIDE A PLOT
    cData = 100*rand(10,1);
    cM = mean(cData);
    cSEM = 2*std(cData)/sqrt(length(cData));
    errorbar(iD,cM,cSEM,'-','Color',[0,0,0],'LineWidth',LineWidth);
    bar(iD,cM,'FaceColor',Colors.(DataByPlot{iP}{iD}),'BarWidth',0.5,'EdgeColor',Colors.(DataByPlot{iP}{iD}));
  end
  set(gca,'XTick',[1:length(DataByPlot{iP})],'XTickLabel',DataByPlot{iP});
  ylim([0,100]); xlim([0.25,iD+0.75]);
  if iP == 1; ylabel('Performance [%]'); end
end
HF_setFigProps('SortMethod','ChildOrder');

%% SAVE FIGURES
HF_viewsave('path',outpath,'name',[name],'view',P.View,'save',P.Save,...
   'format',Opt.FileFormat,'res',Opt.Resolution);

function LF_recomputeData(Recompute,InPath,Name)
  
  Dirs = setgetDirs; Sep = HF_getSep;
  BaseFile = [Dirs.USVClassify,'figures',filesep,Name];
  
  % COLLECT DIFFERENT KINDS OF DATA

  
  save([InPath,Name],'R');