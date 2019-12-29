%% Load Image
clear all; clc;

zigZag = load("Zig-Zag Pattern.txt");
zigZag = zigZag + 1;

Cheetah = imread("cheetah.bmp");
Cheetah = im2double(Cheetah);
[x,y] = size(Cheetah);

%% Cheetah figure features

count = 1;
for i=1:x-7
    for j=1:y-7
        Cheetah_block_dct = dct2(Cheetah(i:i+7,j:j+7));
        Block_dct(zigZag) = Cheetah_block_dct;
        Vectors64D(count,:) = Block_dct;
        count = count + 1;
    end
end

[~,Feature] = sort(abs(Vectors64D),2);
Feature_Cheetah =  Feature(:,63) ;

%% TrainingDataSet features

Data = importdata("TrainingSamplesDCT_8.mat");

Foreground = Data.TrainsampleDCT_FG;
[rowFG, columnFG] = size(Data.TrainsampleDCT_FG);
[~,Features_FG] = sort(abs(Foreground),2);
Feature_FG = Features_FG(:,63);

Background = Data.TrainsampleDCT_BG;
[rowBG, columnBG] = size(Data.TrainsampleDCT_BG);
[~,Features_BG] = sort(abs(Background),2);
Feature_BG = Features_BG(:,63);
%% Plot histograms
figure;
Ch = histogram(Feature_Cheetah);

figure;
Bg = histogram(Feature_BG);
Bg_plot = Bg.Values/rowBG;
Bg_data = zeros(1,64);
[~,cl_bg] = size(Bg_plot);
Bg_data(1,2:cl_bg+1) = Bg_plot;
bar(Bg_data);
set(get(gca,'YLabel'),'String','P(X|grass)','FontSize',10);
set(get(gca,'XLabel'),'String','X(index of 2nd largest)','FontSize',10);

figure;
Fg = histogram(Feature_FG);
Fg_plot = Fg.Values/rowFG;
Fg_data = zeros(1,64);
[~,cl_fg] = size(Fg_plot);
Fg_data(1,2:cl_fg+1) = Fg_plot;
bar(Fg_data);
set(get(gca,'YLabel'),'String','P(X|cheetah)','FontSize',10);
set(get(gca,'XLabel'),'String','X(index of 2nd largest)','FontSize',10);

%% Mask using Training Dataset

prior_ch = rowFG/(rowFG+rowBG);
prior_grass = rowBG/(rowFG+rowBG);

k=1;
for cnt=1:64
    yes = Fg_data(1,cnt)*prior_ch;
    no = Bg_data(1,cnt)*prior_grass;
    if yes>=no
        bin(k) = cnt;
        k=k+1;        
    end
end

Image_result = zeros(255,270);
ginti = 1;
for i=1:x-7
    for j=1:y-7
        if ismember(Feature_Cheetah(ginti),bin)
            Image_result(i,j) = 1;            
        end
        ginti = ginti + 1;
    end
end

figure;
imagesc(Image_result)
colormap(gray(255))

%% Error Calculation

desired = imread("cheetah_mask.bmp");
desired = im2double(desired);
error_matrix = abs(Image_result - desired);
error = sum(error_matrix,"all");
per_error = error/65524*100;
