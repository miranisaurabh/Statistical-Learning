clear;
%% Part (a)
n = 1053 + 250; % total number of independent observations
c_cheetah = 250;
c_grass = 1053;
P_cheetah = c_cheetah/n;
P_grass = c_grass/n;

%% Part (b)
data = importdata("TrainingSamplesDCT_8_new.mat");
TrainFG = data.TrainsampleDCT_FG;
TrainBG = data.TrainsampleDCT_BG;
covFG = cov(TrainFG);
covBG = cov(TrainBG);

meanFG = sum(TrainFG)/c_cheetah;
meanBG = sum(TrainBG)/c_grass;

sigma_cheetah = zeros(64,64);
for i=1:c_cheetah
    temp = TrainFG(i,:) - meanFG;
    sigma_cheetah = sigma_cheetah + temp'*temp;
end
sigma_cheetah = sigma_cheetah/c_cheetah;
var_cheetah = diag(sigma_cheetah)';

sigma_grass = zeros(64,64);
for i=1:c_grass
    temp = TrainBG(i,:) - meanBG;
    sigma_grass = sigma_grass + temp'*temp;
end
sigma_grass = sigma_grass/c_grass;
var_grass = diag(sigma_grass)';

%% Plot
figure;
j=1;
for i=1:64
    

    
    min1 = min(meanFG(i) - 4*sqrt(var_cheetah(i)),meanBG(i) ...
        - 4*sqrt(var_grass(i)));
    max1 = max(meanFG(i) + 4*sqrt(var_cheetah(i)),meanBG(i) ...
        + 4*sqrt(var_grass(i)));

    data_x = linspace(min1,max1,100);
    temp1 = (data_x' - meanFG(i)).^2/var_cheetah(i);
    P_xc = exp(-0.5*log(2*pi)-0.5*log(var_cheetah(i)) -0.5*temp1);
%    P_xc = 1/sqrt((2*pi)^64*var_cheetah(i))*exp(-0.5*temp1);
%     if i<=32
%         subplot(8,4,i)
%     elseif i==33
%         figure;
%         subplot(8,4,i-32)
%     else
%         subplot(8,4,i-32)
%     end
%     plot(data_x', P_xc);
%     hold on;


    

    temp2 = (data_x' - meanBG(i)).^2/var_grass(i);
%    P_xg = exp(-0.5*log(2*pi)-0.5*log(var_grass(i)) -0.5*temp2);
    P_xg = 1/sqrt((2*pi)*var_grass(i))*exp(-0.5*temp2);
    
    if sum(i==[2 3 4 5 59 62 63 64])
        subplot(3,3,j)
        plot(data_x', P_xc);
        hold on;
        plot(data_x', P_xg,'-r')
        title(['Feature ' num2str(i)])
        j = j+1;
    end
    
    
end


%% Part (c)

zigZag = load("Zig-Zag Pattern.txt");
zigZag = zigZag + 1;

Cheetah = imread("cheetah.bmp");
Cheetah = im2double(Cheetah);
[x,y] = size(Cheetah);

% Cheetah figure features
Image_result = zeros(255,270);
count = 1;

Vectors64D = zeros((x-7)*(y-7), 64);
Block_dct = zeros(1,64);
for i=1:x-7
    for j=1:y-7
        Cheetah_block_dct = dct2(Cheetah(i:i+7,j:j+7));
        Block_dct(zigZag) = Cheetah_block_dct;
        Vectors64D(count,:) = Block_dct;
             
        
        d_fg = (Vectors64D(count,:)-meanFG)*((sigma_cheetah)\(Vectors64D(count,:)-meanFG)');
        alpha_fg = log((2*pi)^64*det(sigma_cheetah))-2*log(P_cheetah);
        maha_fg = d_fg + alpha_fg;
        
        d_bg = (Vectors64D(count,:)-meanBG)*((sigma_grass)\(Vectors64D(count,:)-meanBG)');
        alpha_bg = log((2*pi)^64*det(sigma_grass))-2*log(P_grass);
        maha_bg = d_bg + alpha_bg;
        
        count = count + 1;
        
        if maha_fg<=maha_bg
            Image_result(i,j) = 1;
        end
        
    end
end

figure;
imagesc(Image_result)
colormap(gray(255))

 %% 8-D Gaussian

%best_features = [1 2 3 4 5 6 8 26];
%best_features = [1 3 4 59 60 62 63 64];
%best_features = [1 2 3 4 29 50 57 58 ];
best_features = [1 18 25 27 32 41 40 26];
%best_features = [2 3 4 5 59 62 63 64];

TrainFG_8D = TrainFG(:,best_features);
TrainBG_8D = TrainBG(:,best_features);

meanFG_8D = sum(TrainFG_8D)/c_cheetah;
meanBG_8D = sum(TrainBG_8D)/c_grass;

sigma_c8 = zeros(8,8);
for i=1:c_cheetah
    temp = TrainFG_8D(i,:) - meanFG_8D;
    sigma_c8 = sigma_c8 + temp'*temp;
end
sigma_c8 = sigma_c8/c_cheetah;
var_c8 = diag(sigma_c8)';

sigma_g8 = zeros(8,8);
for i=1:c_grass
    temp = TrainBG_8D(i,:) - meanBG_8D;
    sigma_g8 = sigma_g8 + temp'*temp;
end
sigma_g8 = sigma_g8/c_grass;
var_g8 = diag(sigma_g8)';

  
Vectors8D = Vectors64D(:,best_features);

Image_result8 = zeros(255,270);
count = 1;
for i=1:x-7
    for j=1:y-7
                   
        d_fg8 = (Vectors8D(count,:)-meanFG_8D)*((sigma_c8)...
        \(Vectors8D(count,:)-meanFG_8D)');
        alpha_fg8 = log((2*pi)^8*det(sigma_c8))-2*log(P_cheetah);
        maha_fg = d_fg8 + alpha_fg8;
        
        d_bg8 = (Vectors8D(count,:)-meanBG_8D)*((sigma_g8)...
        \(Vectors8D(count,:)-meanBG_8D)');
        alpha_bg8 = log((2*pi)^8*det(sigma_g8))-2*log(P_grass);
        maha_bg = d_bg8 + alpha_bg8;
        
        count = count + 1;
        
        if maha_fg<=maha_bg
            Image_result8(i,j) = 1;
        end
        
    end
end

figure;
imagesc(Image_result8)
colormap(gray(255))
  
% Error
desired = imread("cheetah_mask.bmp");
desired = im2double(desired);
desired_c = sum(desired,'all');
desired_g = 255*270-desired_c;

error_matrix = Image_result8-desired;
error_c = sum(error_matrix==-1,'all');
error_g = sum(error_matrix==1,'all');
error = (error_c/desired_c)*P_cheetah + (error_g/desired_g)*P_grass;
per_error8 = error*100

error_matrix = Image_result-desired;
error_c = sum(error_matrix==-1,'all');
error_g = sum(error_matrix==1,'all');
error = error_c/desired_c*P_cheetah + error_g/desired_g*P_grass;
per_error = error*100
