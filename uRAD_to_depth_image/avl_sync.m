clear;
clc;
close all;

Fs = 200000; 
Fs_CW = 25000;
max_voltage = 3.3;
ADC_bits = 12;
ADC_intervals = 2^ADC_bits;
max_fd = 12500;
mode = 3; 
f0 = 5; 
BW = 240;
Ns = 200;
Ntar = 5; 
Rmax = 100; 
MTI = 0; 
Mth = 1; 
N_FFT = 4096;
c = 299792458;
RampTimeReal = 0.001;
RampTimeReal2 = 0.00075;
factorPresencia_CW = 40;
factorPresencia_FMCW = 22.58;

BW_actual = BW * 1000000;
f0_v = f0*1000000000 + BW_actual/2;
max_velocity = c/(2*f0_v) * max_fd;
max_distance = c/(2*BW_actual) * Fs/2 * RampTimeReal;
distance_vec = linspace(-max_distance, max_distance, N_FFT);
folder_name = "02-07-2025_10-34-15-586727"
folder_path = sprintf("../Downloads/%s/%s/", folder_name, folder_name);
I = readtable(folder_path+"I.csv");
I = I.Variables;
Q = readtable(folder_path+"Q.csv");
Q = Q.Variables;
% time = test(:,801);
I = I(:,2:end);
Q = Q(:,2:end);
images = dir(folder_path+"images/depth/");
images = images(3:end);
conf_images = dir(folder_path+"images/conf/");
conf_images = conf_images(3:end);
sync_idx = readtable(folder_path+"synced/synced_idx.csv");
sync_idx = sync_idx.Variables;

depth_image_means = zeros(1,length(images));
depth_image2_means = zeros(1,length(images));
max_depth = -1;
min_depth = Inf;
mean_Q = [0 0 0];
filter_size = 7;
filter = ones(filter_size) ./ filter_size^2;

for idx = 1:length(images)
    depth_I = imread(sprintf(folder_path+"images/depth/%s",images(idx).name));
    depth_I = depth_I(:,26:end);
    depth_I2 = conv2(depth_I, filter, "same");
    depth_image_means(idx) = mean(mean(depth_I));
    depth_image2_means(idx) = mean(mean(depth_I2));
    min_depth = min(min_depth, min(min(depth_I)));
    max_depth = max(max_depth,max(max(depth_I)));
    sort_I = sort(depth_I(:));
    Q1_idx = length(sort_I) * 0.25;
    Q2_idx = length(sort_I) * 0.5;
    Q3_idx = length(sort_I) * 0.75;
    mean_Q = mean_Q + [sort_I(Q1_idx) sort_I(Q2_idx) sort_I(Q3_idx)];
end
mean_freq = zeros(1,4096);
mean_Q = mean_Q ./ length(images);
heat_map = zeros(size(depth_I));
for idx = 1:length(I)
    figure(1);
    test_q = Q(idx,:);
    test_i = I(idx,:);
    
    maxVoltage = 3.3;
    ADC_intervals = 2^12;
    temp_i = test_i * maxVoltage / ADC_intervals;
    temp_i = temp_i - mean(temp_i);
    
    temp_q = test_q * maxVoltage / ADC_intervals;
    temp_q = temp_q - mean(temp_q);

    complex_vec1 = temp_i + i*temp_q;

    complex_vec1 = complex_vec1 .* hanning(Ns)' .* 2 / 3.3;

    FreqDomain1 = 2 * abs(fftshift(fft(complex_vec1/ Ns, N_FFT)));

    start = round(N_FFT/2);

    FreqDomain1(start) = FreqDomain1(start-1);

    FreqDomain1 = 20 * log10(FreqDomain1);
    mean_freq = mean_freq + FreqDomain1;
    temp_freq1 = (FreqDomain1 - min(FreqDomain1)) ./ (max(FreqDomain1) - min(FreqDomain1));
    subplot(1,3,1);
    plot(FreqDomain1(2048:2548));
    [~, max_idx] = max(FreqDomain1);
    tempTemp = sprintf("d = %f m", distance_vec(max_idx));
    title(tempTemp);
    ylim([-100, 0])
    subplot(1,3,2);
    depth_I = imread(sprintf(folder_path+"images/depth/%s",images(sync_idx(idx,3)+1).name));
    depth_I = depth_I(:,26:end);
    depth_I(depth_I < mean_Q(1)*0.5) = 2200;
    depth_I(depth_I > mean_Q(3)) = 2200;
    depth_I2 = conv2(depth_I, filter, "same");
    % imagesc(depth_I);
    imshow(depth_I2, [min_depth max_depth]);
    colorbar;
    subplot(1,3,3);
    % conf_I = imread(sprintf(folder_path+"images/conf/%s",conf_images(sync_idx(idx,3)+1).name));
    % if distance_vec(max_idx) < 2
    % heat_map = heat_map + (abs(depth_I - distance_vec(max_idx)*1000) < 200);
    % end
    % imshow(heat_map,[])
    % imagesc(conf_I);
    % hist(depth_I2);
    depth_snippet = depth_I2(63-32:63+31, 83-32:83+31);
    imshow(depth_snippet,[min_depth max_depth]);
    drawnow;
    pause(0.016666);

end
mean_freq = mean_freq ./ length(I);