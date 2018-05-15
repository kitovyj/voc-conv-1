function [MidDist,SCorr] = VocLocalizer(varargin)
% COLLECT PROPERTIES AND PERFORM INITIAL ANALYSIS OF VOCALIZATIONS  

P = parsePairs(varargin);
checkField(P,'Sounds')
checkField(P,'SR')
checkField(P,'NFFT',1000);
checkField(P,'Offset',1);
checkField(P,'HighPass',25000);
checkField(P,'CorrTime',0.0001); % 0.1 ms
checkField(P,'CenterShift',0.01); %10mm of CenterShift based on measured discrepancy between position of left and right microphone
checkField(P,'LocMethod','Empirical');
checkField(P,'RemoveAmplitude',1);
checkField(P,'CorrMethod','GCC');
checkField(P,'DelayRange',0.000075); %75us
checkField(P,'FreqSelect',1.5);
checkField(P,'FIG',1); % 

% POTENTIALLY FILTER THE DATA
if P.HighPass
  [b,a]=butter(4,P.HighPass/(P.SR/2),'high');
  for i=1:length(P.Sounds) 
    P.Sounds{i} = filter(b,a,P.Sounds{i}); 
  end
end

if P.RemoveAmplitude
  for i=1:length(P.Sounds)
    H{i} = abs(hilbert(P.Sounds{i}));
    P.Sounds{i} = P.Sounds{i}./H{i}; 
  end
  %R = xcorr(H{1}/max(H{1}),H{2}/max(H{2}),100,'unbiased');
  %max(R)
end


CorrSteps = round(P.CorrTime*P.SR);
switch P.CorrMethod
  case 'Specgram';
    % COMPUTE SPECTROGRAM OF THE SOUND FROM TWO MICROPHONES
    S1 = HF_specgram(P.Sounds{1},P.NFFT,P.SR,[],P.NFFT-P.Offset,0,0);
    S2 = HF_specgram(P.Sounds{2},P.NFFT,P.SR,[],P.NFFT-P.Offset,0,0);
    
    % SELECT FREQUENCY INDICES TO COMPUTE THE CROSSCORRELATION
    FreqMarginals = mean(abs(S1)+abs(S2),2);
    FreqInds = find( FreqMarginals > P.FreqSelect*median(FreqMarginals));
    disp(num2str(length(FreqInds)))
    
    Corrs= zeros(length(FreqInds),2*CorrSteps+1);
    
    % COMPUTE CROSSCORRELATIONS
    for iF=1:length(FreqInds)
      cInd = FreqInds(iF);
      Corrs(iF,:) = xcorr(real(S1(cInd,:)),real(S2(cInd,:)),CorrSteps,'unbiased');
      CorrsA(iF,:) = xcorr(abs(S1(cInd,:)),abs(S2(cInd,:)),CorrSteps,'unbiased');
    end
    
    Corrs = Corrs./repmat(FreqMarginals(FreqInds),1,size(Corrs,2));
    SCorr = abs(hilbert(mean(Corrs)));
        
  case 'GCC';
    NFFT = length(P.Sounds{1});
    S1F = fft(P.Sounds{1},NFFT); S2F = fft(P.Sounds{2},NFFT);
    FLow = 40000; FHigh = 125000;
    F = [1:NFFT]/NFFT*P.SR; F = F(1:round(NFFT/2));
    ind = (F<FLow).*(F>FHigh);
    ind = logical([ind,fliplr(ind)]);
    S1F(ind) = 0; S2(ind) = 0;
    R = S1F.*conj(S2F)./(abs(S1F).*abs(conj(S2F)));
    XCorr = ifft(R);
    SCorr = abs(hilbert([XCorr(round(NFFT/2)+1:end);XCorr(1:round(NFFT/2))]));
    SCorr = SCorr(round(end/2)-CorrSteps:round(end/2)+CorrSteps);
  case 'XCorr';
    SCorr = xcorr(P.Sounds{1},P.Sounds{2},CorrSteps,'unbiased');
end
Time = [-CorrSteps:CorrSteps]/P.SR;
NotInd = find(abs(Time)>P.DelayRange);
SCorr(NotInd) = -inf;
[M,Pos] = max(SCorr);

DeltaTime = real((Pos-(CorrSteps+1))/P.SR); % DeltatTime is negative, if signal arrives first on the left (channel 1), i.e. position is also on the left

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


