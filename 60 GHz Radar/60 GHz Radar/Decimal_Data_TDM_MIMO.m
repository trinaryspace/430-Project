
%MIMO Data Extraction with Time Divison Multiplexing (TDM MIMO)

function [M] = Decimal_Data_TDM_MIMO(channel,chirp,frame, adcsample, adcsamplerate, address, TX)

fs=adcsamplerate; %sampling frequency
fid = fopen(address,'r');
adcdata1 = fread(fid, 'int16');

%fread.close();

%creating a vector of adc data from new decimal integers 
adcdata2 = transpose(adcdata1);

%creating a 5D matrix called M in which for each frame, and each chirp, there are four channels which we put the adc samples(256) in each of them
LSB=1;
for L=1:1:TX
    for i=1:1:frame
        for j=1:1:chirp
            for k=1:1:channel
                
                q= (j-1)*TX + L;
                a=1+(chirp*channel*(i-1)+channel*(q-1)+(k-1))*adcsample;
                b=adcsample+(chirp*channel*(i-1)+channel*(q-1)+(k-1))*adcsample;
                M(i,j,k,:,L)=LSB*adcdata2(a:b); 
                
            end
        end
    end
end

end


