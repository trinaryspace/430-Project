

function [AOAFFT_fftshift_Azimuth, angle_Azimuth, AOAFFT_fftshift_elevation_4, angle_elevation] = AoAFFT_TDM_MIMO(M, channel,chirp, adcsample, adcsamplerate, ...
    framenumber, Slope, Tc, LSB, Npadding, lambda, TX)

for i=1:1:TX
    for j=1:1:channel
        channelnumber= j;
        [P,~] = RangeFFT(M(:,:,:,:,i),framenumber, channelnumber, Slope, Tc, LSB, chirp, adcsample);
        Averaged_RangeFFT(j,i,:)= mean((P(:,1:end/2))/(adcsamplerate/adcsample),1);
    end
end
Q_Azimuth= reshape([Averaged_RangeFFT(1,1,:); Averaged_RangeFFT(2,1,:); Averaged_RangeFFT(3,1,:); Averaged_RangeFFT(4,1,:); Averaged_RangeFFT(1,3,:);Averaged_RangeFFT(2,3,:);Averaged_RangeFFT(3,3,:);Averaged_RangeFFT(4,3,:);Averaged_RangeFFT(1,4,:);Averaged_RangeFFT(2,4,:);Averaged_RangeFFT(3,4,:);Averaged_RangeFFT(4,4,:)], [12,adcsample/2]);
Q_elevation_1= reshape([Averaged_RangeFFT(3,1,:); Averaged_RangeFFT(1,2,:)],[2,adcsample/2]);
Q_elevation_2= reshape([Averaged_RangeFFT(4,1,:); Averaged_RangeFFT(2,2,:)],[2,adcsample/2]);
Q_elevation_3= reshape([Averaged_RangeFFT(1,3,:); Averaged_RangeFFT(3,2,:)],[2,adcsample/2]);
Q_elevation_4= reshape([Averaged_RangeFFT(2,3,:); Averaged_RangeFFT(4,2,:)],[2,adcsample/2]);

Q_Azimuth_Zero_padding= [Q_Azimuth; zeros(Npadding-3*4,adcsample/2)];
Q_elevation_1_Zero_padding= [Q_elevation_1; zeros(Npadding-2,adcsample/2)];
Q_elevation_2_Zero_padding= [Q_elevation_2; zeros(Npadding-2,adcsample/2)];
Q_elevation_3_Zero_padding= [Q_elevation_3; zeros(Npadding-2,adcsample/2)];
Q_elevation_4_Zero_padding= [Q_elevation_4; zeros(Npadding-2,adcsample/2)];

%need to change
d_azimuth=lambda/2;
d_elevation= 0.8*lambda;

omega_azimuth = linspace(-pi,pi, Npadding);
omega_elevation = linspace(-2*pi*0.8,2*pi*0.8, Npadding);

angle_Azimuth = asin(omega_azimuth/pi)*180/pi;
angle_elevation = asin(omega_elevation/pi)*180/pi;
% theta_max_azimuth= asin(lambda/2/d_azimuth); 
% angle= linspace(-theta_max, theta_max, Npadding)*180/pi; 

for j=1:1:adcsample/2
AOAFFT_Azimuth(:,j)= fft(Q_Azimuth_Zero_padding(:,j));
AOAFFT_fftshift_Azimuth(:,j)= fftshift(AOAFFT_Azimuth(:,j));

AOAFFT_elevation_1(:,j)= fft(Q_elevation_1_Zero_padding(:,j));
AOAFFT_fftshift_elevation_1(:,j)= fftshift(AOAFFT_elevation_1(:,j));

AOAFFT_elevation_2(:,j)= fft(Q_elevation_2_Zero_padding(:,j));
AOAFFT_fftshift_elevation_2(:,j)= fftshift(AOAFFT_elevation_2(:,j));

AOAFFT_elevation_3(:,j)= fft(Q_elevation_3_Zero_padding(:,j));
AOAFFT_fftshift_elevation_3(:,j)= fftshift(AOAFFT_elevation_3(:,j));

AOAFFT_elevation_4(:,j)= fft(Q_elevation_4_Zero_padding(:,j));
AOAFFT_fftshift_elevation_4(:,j)= fftshift(AOAFFT_elevation_4(:,j));
end

AOAFFT_fftshift_elevation= (AOAFFT_fftshift_elevation_1+AOAFFT_fftshift_elevation_2+AOAFFT_fftshift_elevation_3+AOAFFT_fftshift_elevation_4)/4;

end
