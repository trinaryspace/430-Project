

function [P,distance] = RangeFFT(M,framenumber, channelnumber, Slope, Tc, LSB,chirp, adcsample)

N=reshape(M(framenumber, :, channelnumber, :), chirp, adcsample); %%converting 4D to 2D
window1DFFT=transpose(hann(adcsample));
% % Bandwidth= Slope*Tc;

% Bandwidth= 2.0566e9;%150m
% Bandwidth= 2.2544e9;%125m
% Bandwidth= 3.065e9;%100m
Bandwidth= 4.1924e9; %75m
% Bandwidth= 4.3174e9;

dres=3e8/(2*Bandwidth);
distance0=dres*(adcsample/2);
% distance0=100;
distance=linspace(0,distance0,adcsample/2);
window1DFFT=transpose(hann(adcsample));

for i=1:1:chirp
    P(i,:)= fft(LSB*N(i,:).*window1DFFT);
end

end
