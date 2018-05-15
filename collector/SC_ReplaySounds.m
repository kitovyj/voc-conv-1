function [RNoise, RVocal,SNoise, SVocal] = SC_ReplaySounds

Positions = [-0.05:0.005:0.05]; 
clear RVocal RNoise; 

for iP=1:length(Positions)
  fprintf(['Position : ',num2str(Positions(iP)*1000),'mm']); 
  pause;
  fprintf('\n');
  for iR=1:10   
    [R,S] = ReplaySounds; 
    RNoise(:,:,iR,iP) = R; 
    if iP==1 && iR==1 SNoise = S; end; 
    [R,S] = ReplaySounds('Stimulus','Vocal','Duration',0.2); 
    RVocal(:,:,iR,iP) = R; 
    if iP==1 && iR==1 SVocal = S; end; 
  end; 
end;