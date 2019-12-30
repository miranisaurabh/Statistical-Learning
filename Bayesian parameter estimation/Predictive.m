%% Initialization

clear;
tic
data = importdata("TrainingSamplesDCT_subsets_8.mat");
TrainFG = data.D1_FG;
TrainBG = data.D1_BG;

c_cheetah = size(TrainFG,1);
c_grass = size(TrainBG,1);
n = c_cheetah + c_grass;

P_cheetah = c_cheetah/n;
P_grass = c_grass/n;
%covFG = cov(TrainFG);
%covBG = cov(TrainBG);

meanFG = (sum(TrainFG)/c_cheetah)';
meanBG = (sum(TrainBG)/c_grass)';

sigma_cheetah = zeros(64,64);
for i=1:c_cheetah
    temp = TrainFG(i,:) - meanFG';
    sigma_cheetah = sigma_cheetah + temp'*temp;
end
sigma_cheetah = sigma_cheetah/c_cheetah;


sigma_grass = zeros(64,64);
for i=1:c_grass
    temp = TrainBG(i,:) - meanBG';
    sigma_grass = sigma_grass + temp'*temp;
end
sigma_grass = sigma_grass/c_grass;

prior_data = importdata("Prior_1.mat");
w0_i = prior_data.W0;
alpha_data = importdata("Alpha.mat");

desired = imread("cheetah_mask.bmp");
desired = im2double(desired);
desired_c = sum(desired,'all');
desired_g = 255*270-desired_c;

% Execute only once hence outside loop
per_error_pred = zeros(1,9);
per_error_ml = zeros(1,9);
per_error_MAP = zeros(1,9);

zigZag = load("Zig-Zag Pattern.txt");
zigZag = zigZag + 1;

Cheetah = imread("cheetah.bmp");
Cheetah = im2double(Cheetah);
[x,y] = size(Cheetah);
Vectors64D = zeros((x-7)*(y-7), 64);
Block_dct = zeros(1,64);

%% Compute

for k=9:9
    
    
    alpha = alpha_data(k); %Using only 1 value of alpha as of now
    
    var0 = alpha*w0_i;
    sigma_0 = diag(var0);
    
    beta1_cheetah = sigma_0/(sigma_0 + (sigma_cheetah/c_cheetah));
    beta2_cheetah = (sigma_cheetah/(sigma_0 + (sigma_cheetah/c_cheetah)))...
        /c_cheetah;
    
    beta1_grass = sigma_0/(sigma_0 + (sigma_grass/c_grass));
    beta2_grass = (sigma_grass/(sigma_0 + (sigma_grass/c_grass)))/c_grass;
    
    mu0_cheetah = prior_data.mu0_FG';
    mu0_grass = prior_data.mu0_BG';
    
    bayes_mu_cheetah =  beta1_cheetah*meanFG + beta2_cheetah*mu0_cheetah;
    bayes_mu_grass = beta1_grass*meanBG + beta2_grass*mu0_grass;
    
    bayes_sigma_cheetah = (beta1_cheetah*sigma_cheetah)/c_cheetah;
    bayes_sigma_grass = (beta1_grass*sigma_grass)/c_grass;
    
    predictive_mean_cheetah = bayes_mu_cheetah;
    predictive_mean_grass = bayes_mu_grass;
    
    predictive_sigma_cheetah = sigma_cheetah + bayes_sigma_cheetah;
    predictive_sigma_grass = sigma_grass + bayes_sigma_grass;
    
    
    

    
    % Cheetah figure features
    Image_result_pred = zeros(255,270);
    Image_result_ml = zeros(255,270);
    Image_result_MAP = zeros(255,270);
    count = 1;
    
    
    for i=1:x-7
        for j=1:y-7
            
            if k==9  
                Cheetah_block_dct = dct2(Cheetah(i:i+7,j:j+7));
                Block_dct(zigZag) = Cheetah_block_dct;
                Vectors64D(count,:) = Block_dct;            
            end
            
            % Predicitive Distribution
            d_fg = (Vectors64D(count,:)-predictive_mean_cheetah')...
                *((predictive_sigma_cheetah)...
                \(Vectors64D(count,:)-predictive_mean_cheetah')');
            alpha_fg = log((2*pi)^64*det(predictive_sigma_cheetah))...
                -2*log(P_cheetah);
            maha_fg = d_fg + alpha_fg;
            
            d_bg = (Vectors64D(count,:)-predictive_mean_grass')...
                *((predictive_sigma_grass)...
                \(Vectors64D(count,:)-predictive_mean_grass')');
            alpha_bg = log((2*pi)^64*det(predictive_sigma_grass))-...
                2*log(P_grass);
            maha_bg = d_bg + alpha_bg;            
            
            if maha_fg<=maha_bg
                Image_result_pred(i,j) = 1;
            end
            
            
            % ML estimate            
            d_fg = (Vectors64D(count,:)-meanFG')...
                *((sigma_cheetah)\(Vectors64D(count,:)-meanFG')');
            alpha_fg = log((2*pi)^64*det(sigma_cheetah))-2*log(P_cheetah);
            maha_fg = d_fg + alpha_fg;
            
            d_bg = (Vectors64D(count,:)-meanBG')*((sigma_grass)...
                \(Vectors64D(count,:)-meanBG')');
            alpha_bg = log((2*pi)^64*det(sigma_grass))-2*log(P_grass);
            maha_bg = d_bg + alpha_bg;
            
            if maha_fg<=maha_bg
                Image_result_ml(i,j) = 1;
            end
            
            % MAP estimate
            d_fg = (Vectors64D(count,:)-predictive_mean_cheetah')...
                *((sigma_cheetah)...
                \(Vectors64D(count,:)-predictive_mean_cheetah')');
            alpha_fg = log((2*pi)^64*det(sigma_cheetah))-2*log(P_cheetah);
            maha_fg = d_fg + alpha_fg;
            
            d_bg = (Vectors64D(count,:)-predictive_mean_grass')...
                *((sigma_grass)...
                \(Vectors64D(count,:)-predictive_mean_grass')');
            alpha_bg = log((2*pi)^64*det(sigma_grass))-2*log(P_grass);
            maha_bg = d_bg + alpha_bg;
            
            if maha_fg<=maha_bg
                Image_result_MAP(i,j) = 1;
            end
            
           
            count = count + 1;
            
        end
    end
    
    %figure;
    %imagesc(Image_result)
    %colormap(gray(255))
    
    error_matrix = Image_result_pred-desired;
    error_c = sum(error_matrix==-1,'all');
    error_g = sum(error_matrix==1,'all');
    error = (error_c/desired_c)*P_cheetah + (error_g/desired_g)*P_grass;
    per_error_pred(k) = error;
    
    error_matrix = Image_result_ml-desired;
    error_c = sum(error_matrix==-1,'all');
    error_g = sum(error_matrix==1,'all');
    error = (error_c/desired_c)*P_cheetah + (error_g/desired_g)*P_grass;
    per_error_ml(k) = error;
    
    error_matrix = Image_result_MAP-desired;
    error_c = sum(error_matrix==-1,'all');
    error_g = sum(error_matrix==1,'all');
    error = (error_c/desired_c)*P_cheetah + (error_g/desired_g)*P_grass;
    per_error_MAP(k) = error;
    
    disp(k)
    disp(per_error_pred(k))
    disp(per_error_ml(k))
    disp(per_error_MAP(k))
    toc
end

figure;
semilogx(alpha_data,per_error_pred,'DisplayName','Predictive');
hold on;
semilogx(alpha_data,per_error_ml,'DisplayName','ML');
semilogx(alpha_data,per_error_MAP,'DisplayName','MAP');
hold off;

xlabel('alpha')
ylabel('PoE')
title('PoE v/s alpha: Dataset:1 Strategy:1')
legend('Location','southeast')




