clc
clear all
close all
warning('off', 'all'); % Suppress all warnings

c=3e8;
f=76.5e9;
lambda= c/f;
channel=4;
chirp=128;
TX=3;
frame=8;
 % adcsample= 1440;
adcsample=4096;
adcsamplerate=100e5; %% in Hz, 10MHz
fs= adcsamplerate;
address= "C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin";
% address= "D:\MTT\SNR Measurement\outdoor\Setup\background.bin";
% address= "D:\MTT\SNR Measurement\outdoor\background.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\MTT\SNR Measurement\SNR January 29\a.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\SIW Modulated\January
% 25th\With_WaveformGenerator_4MHz.bin";`
% address= "C:\Users\beamd\Desktop\Sepideh\SIW Modulated\January_19th\environment.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\SIW Modulated\3pair\0_0v.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\MTT\SNR Measurement\SNR_Data_highet_R_max\2_72m.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\MTT\SNR Measurement\SNR Data\empty.bin";
% address= "D:\JRFID paper\JRFID Final Measurements\JRFID Signal processing Measurements\indoor\Single.bin";
% address= "C:\Users\beamd\Desktop\Sepideh\JRFID Final Measurements\outdoor\SNR_vs_range\11m.bin";
[M] = Decimal_Data_TDM_MIMO(channel,chirp,frame, adcsample, adcsamplerate, address, TX);

% Bounds
meter=20;
d_low= meter-4*0.8;  
d_up= meter+4*0.8;
% 
% d_low= 0;
% d_up= 130;

angel_SNR =0;

framenumber=1; 
channelnumber= 1;
chirpnumber=1;
% 
% Slope= 29.982e12; %25m
% Tc= (150-6)*1e-6;

Slope= 10.235e12;%75m
Tc= (430-6)*1e-6;

% 

% Slope= 5.504e12;%125m
% Tc= (800-6)*1e-6;


% Slope= 7.483e12;%100m
% Tc= (600-6)*1e-6;

% Slope= 5.021e12; %150m
% Tc= (850-6)*1e-6;

LSB=1;

for i=1:1:TX
[P(:,:,i),distance] = RangeFFT(M(:,:,:,:,i),framenumber, channelnumber, Slope, Tc, LSB, chirp, adcsample);
Averaged_RangeFFT(:,i)= mean((P(:,[1:end/2],i))/(adcsamplerate/adcsample),1);
end

figure(1)
plot(distance, 20*log10(abs(Averaged_RangeFFT(:,1))), 'LineWidth', 1)
% plot(distance, abs(sum(Averaged_RangeFFT,2)).^2, 'LineWidth', 1)
grid on
title([' Averaged Range FFT '],'FontName','Times')
ylabel(' Magnitude','FontName','Times');
xlabel(' Range [m] ','FontName','Times');
% d_low= 0.2;
% d_up= 80;
xlim([d_low, d_up])

Npadding= 2^8;
[AOAFFT_fftshift_Azimuth, angle_Azimuth, AOAFFT_fftshift_elevation, angle_elevation] = AoAFFT_TDM_MIMO(M, channel,chirp, adcsample, adcsamplerate, ...
    framenumber, Slope, Tc, LSB, Npadding, lambda, TX);

f1=figure(2);
imagesc(distance, angle_Azimuth, 20*log10(abs(AOAFFT_fftshift_Azimuth)))
% imagesc(distance, angle_Azimuth, abs(AOAFFT_fftshift_Azimuth).^2)
title(' 2D FFT ','FontName','Times')
ylabel(' Azimuth Angle [deg]','FontName','Times','FontSize', 24);
xlabel(' Range [m] ','FontName','Times','FontSize', 24);
xlim([d_low, d_up])
clim([0,0.5])
colormap jet
colorbar
set(gca, 'FontName', 'Times', 'FontSize', 24);
grid on;
% ax=gca;
% ax.PlotBoxAspectRatio = [1.6 1 1];
% f1.Position = [100 90 1160 610];

% print(gcf, 'D:\JRFID paper\JRFID Final Measurements\Figures\pattern_20cm.png','-dpng','-r800');
% print(gcf, 'D:\JRFID paper\JRFID Final Measurements\Figures\pattern_36cm.png','-dpng','-r800');
% print(gcf, 'D:\JRFID paper\JRFID Final Measurements\Figures\pattern_50cm.png','-dpng','-r800');


figure(3)
imagesc(distance, angle_Azimuth, 20*log10(abs(AOAFFT_fftshift_Azimuth)))
% imagesc(distance, angle_elevation, abs(AOAFFT_fftshift_elevation).^2)
title([' Averaged Elevation AoA FFT '],'FontName','Times')
ylabel(' Elevation Angle [deg]','FontName','Times');
xlabel(' Range [m] ','FontName','Times');
xlim([d_low, d_up])
% clim([0,8])
colormap jet
colorbar



[~, indx_angle_SNRl] = min(abs(angle_Azimuth+angel_SNR));
[~, indx_angle_SNRu] = min(abs(angle_Azimuth-angel_SNR));
% Find column indices where the value d_low and d_up appears in any row
[row_indices1, col_indices1] = min(abs(distance-d_low));
[row_indices2, col_indices2] = min(abs(distance-d_up));

% Unique column indices where the value d_low and d_up appears
unique_columns1 = unique(col_indices1);
unique_columns2 = unique(col_indices2);

% Define the column bounds 
col_start = unique_columns1; % Start of the column range
col_end = unique_columns2;   % End of the column range

% Extract the submatrix within the specified column bounds
submatrix = 20*log10(abs(AOAFFT_fftshift_Azimuth(indx_angle_SNRl:indx_angle_SNRu, col_start:col_end)));

% Find the maximum value and its index within the submatrix
[max_value, linear_index] = max(submatrix(:));

% Convert the linear index in the submatrix to row and column indices
[sub_row, sub_col] = ind2sub(size(submatrix), linear_index);

% Map the submatrix column index back to the original matrix column index
original_col = col_start + sub_col - 1;
original_row = sub_row + indx_angle_SNRl -1;
% Display results
disp(['Peak AoA FFT: ', num2str(max_value)]);
disp(['Peak Distance: ', num2str(distance(original_col))]);
disp(['Peak AoA: ', num2str(angle_Azimuth(original_row))]);
% 
% range= 4.57; %vatar
% normal_distance= 0.673;

% AoA= asin(normal_distance/range)*180/pi;


figure(4)
selected_angle= angle_Azimuth(original_row);
[~, selected_angle_index]=min(abs(selected_angle-angle_Azimuth));
pl =plot(distance, 10*log10(abs(AOAFFT_fftshift_Azimuth(selected_angle_index,:)).^2),'DisplayName',['\theta =' num2str(selected_angle)], 'LineWidth', 1);
mean_dB = mean(10*log10(abs(AOAFFT_fftshift_Azimuth(selected_angle_index,:)).^2));
title([' Averaged 2D FFT for specific angle'],'FontName','Times');
ylabel(' Magnitude','FontName','Times');
xlabel(' Range [m] ','FontName','Times');
% ylim([-80 40])
xlim([d_low, d_up])
legend(pl);

% Display results
% AoA = (abs(AOAFFT_fftshift_Azimuth(selected_angle_index,:)).^2);
% [P_sig Pmax_indx]   = max(AoA(1:164));

% del_ind = 20;
% P_noise = min([AoA(Pmax_indx-del_ind:Pmax_indx) AoA(Pmax_indx:Pmax_indx+del_ind)]); % for outdoor
% P_noise = min([AoA(1:164)]); % for indoor
% figure
% plot(10*log10(AoA))
% SNR = 10*log10(AoA(60)/AoA(187));
% AoA(171)
% disp('================================');
% disp(['SNR: ', num2str(SNR)]);



fclose ('all');