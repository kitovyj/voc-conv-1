function [MidDist, SCorr,CorrTime] = VocLocalizerI(varargin)
% COLLECT PROPERTIES AND PERFORM INITIAL ANALYSIS OF VOCALIZATIONS  

P = parsePairs(varargin);
checkField(P,'Sounds');
checkField(P,'Mu',0.02);
checkField(P,'HighPass',0)
checkField(P,'ImpDur',0.01); % ms
checkField(P,'CenterShift',0.009); %10mm of CenterShift based on measured discrepancy between position of left and right microphone
checkField(P,'CorrTime',0.0001); 
checkField(P,'LocMethod','Empirical');
checkField(P,'SR',250000);
checkField(P,'FIG',0); % 

M = round(P.ImpDur*P.SR);
NSteps = length(P.Sounds{1});
NReps = 10;
NIter = NSteps-M+1;
U = zeros(2*M,NReps*NIter); %U(round(4*M/3)) = 1; 
U(:,1) = randn(2*M,1); U(:,1) = U(:,1)/sqrt(sum(U(:,1).^2));
Error = zeros(1,NReps*NIter);

% POTENTIALLY FILTER THE DATA
if P.HighPass
  [b,a]=butter(4,[P.HighPass]/(P.SR/2),'high');
  for i=1:length(P.Sounds) 
    P.Sounds{i} = filter(b,a,P.Sounds{i}); 
  end
end

it = 1;
for k=1:NReps
  for n=1:NIter
    x = [P.Sounds{1}(n:n+M-1);P.Sounds{2}(n:n+M-1)];
    u = U(:,it);
    Error(it) = u'*x;
    U(:,it+1) = u - P.Mu*Error(it)*x;
    U(:,it+1) = U(:,it+1)/sqrt(sum(U(:,it+1).^2));
    it = it +1;
  end
end

R.G(1,:) = U(1:M,it);
R.G(2,:) = -U(M+1:end,it);

for i=1:2
  H(i,:) = abs(hilbert(R.G(i,:)/max(abs(R.G(i,:)))));
  [MAX,Pos(i)] = max(H(i,:));
end
DeltaTime = (Pos(2)-Pos(1))/P.SR
SCorr = R.G(1,:) + R.G(2,:);

CorrTime = [-M/2:-1,1:M/2]/P.SR;

% COMPUTE POSITION FROM TIMING
VSound = 340.29; %m/s
DeltaDist = VSound * DeltaTime;
switch P.LocMethod
  case 'Geometric';
    % COMPUTE POSITION BASED ON SETUP GEOMETRY AND CORRELATION
    MicDist = 0.46; %m % Measurements from new setup
    % Estimate of previous distances : 0.38
    MicHeight = 0.354; %m % Measurements from new setup
    %MicHeight = 0.25; %m % Measurements from new setup
    % Estimate of previous height: 0.35
    D = DeltaDist;
    H = MicHeight;
    S  = MicDist;
    MidDist = real((D*sqrt(D^2-4*H^2-S^2))/sqrt(4*D^2-4*S^2)) - P.CenterShift;
  case 'Empirical';
    MidDist = DeltaDist - P.CenterShift;
end

if P.FIG
  figure(P.FIG); clf; hold on;
 % [DC,AH] = axesDivide(2,1,'c')
  %axes(AH(1));
  Colors = {'b','r'};
  for i=1:2
   plot(H(i,:),'Color',Colors{i});
   plot([Pos(i),Pos(i)],[0,1.2],'Color',Colors{i})
  end
  axis([0,M,0,1.2]);
  title(num2str(Pos(2)-Pos(1)));
  grid on;    
end