clear;
%% Computations

data = importdata("TrainingSamplesDCT_8_new.mat");
TrainFG = data.TrainsampleDCT_FG;
TrainBG = data.TrainsampleDCT_BG;
%TrainBG = TrainFG;
c_cheetah = size(TrainFG,1);
c_grass = size(TrainBG,1);
total_n = c_cheetah + c_grass;
P_cheetah = c_cheetah/total_n;
P_grass = c_grass/total_n;

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

%% Training Gaussian mixtures

max_iterations = 500;
%n_components = 32;
%list_components = [1 2 4 8 16 32];
list_components = [8];
num_mixtures =1;
list_dimensions = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64];
%list_dimensions = 2;

figure;
title('Probability of Error vs. Dimension')
xlabel('No. of dimensions')
ylabel('PoE')

for compo = 1:size(list_components,2)
    
    disp(['Component_id = ',num2str(compo)])
    
    n_components = list_components(compo);
    
    mixture_sigma_FG = containers.Map;
    mixture_mu_FG = containers.Map;
    mixture_pi_FG = containers.Map;
    mixture_sigma_BG = containers.Map;
    mixture_mu_BG = containers.Map;
    mixture_pi_BG = containers.Map;
    
    for id1_mix = 1:num_mixtures
        
        mu_init_bg = zeros(1,64,n_components);
        sigma_init_bg = zeros(64,64,n_components);
        
        mu_init_fg = zeros(1,64,n_components);
        sigma_init_fg = zeros(64,64,n_components);
        
        for i=1:n_components
            
            temp1 = normrnd(5,0.1,[1 64]);
            mu_init_bg(:,:,i) = temp1/sum(temp1);
            
            temp2 = normrnd(5,0.05,[1 64])+5;
            sigma_init_bg(:,:,i) = diag(temp2/sum(temp2));
            
            temp1 = normrnd(5,0.3,[1 64]);
            mu_init_fg(:,:,i) = temp1/sum(temp1);
            
            temp2 = normrnd(5,0.3,[1 64])+5;
            sigma_init_fg(:,:,i) = diag(temp2/sum(temp2));
            
        end
        % pi_init_bg = rand([1,n_components]);
        % pi_init_bg = pi_init_bg/sum(pi_init_bg);
        pi_init_bg = zeros(1,n_components);
        pi_init_bg = pi_init_bg + 1/n_components;
        pi_init_fg = pi_init_bg;
        
        hij_bg = zeros(c_grass,n_components);
        mu_next_bg = zeros(1,64,n_components);
        sigma_next_bg = zeros(64,64,n_components);
        pi_next_bg = zeros(1,n_components);
        
        hij_fg = zeros(c_cheetah,n_components);
        mu_next_fg = zeros(1,64,n_components);
        sigma_next_fg = zeros(64,64,n_components);
        pi_next_fg = zeros(1,n_components);
        
        for iteration=1:max_iterations
            
            for i=1:n_components
                hij_bg(:,i) = multi_gaussian(TrainBG,mu_init_bg(:,:,i),sigma_init_bg(:,:,i))*pi_init_bg(i);
                hij_fg(:,i) = multi_gaussian(TrainFG,mu_init_fg(:,:,i),sigma_init_fg(:,:,i))*pi_init_fg(i);
            end
            
            sumj_hij_bg = sum(hij_bg,2);
            hij_bg = hij_bg./sumj_hij_bg;
            sumi_hij_bg = sum(hij_bg,1);
            
            sumj_hij_fg = sum(hij_fg,2);
            hij_fg = hij_fg./sumj_hij_fg;
            sumi_hij_fg = sum(hij_fg,1);
            
            for i=1:n_components
                
                % Mu next
                numerator_mu = sum(hij_bg(:,i).*TrainBG);
                mu_next_bg(:,:,i) = numerator_mu/sumi_hij_bg(i);
                
                numerator_mu = sum(hij_fg(:,i).*TrainFG);
                mu_next_fg(:,:,i) = numerator_mu/sumi_hij_fg(i);
                
                % Pi next
                pi_next_bg(i) = sumi_hij_bg(i)/c_grass;
                
                pi_next_fg(i) = sumi_hij_fg(i)/c_cheetah;
                
                % Sigma next
                tmp_sigma = sum(hij_bg(:,i).*((TrainBG - mu_init_bg(:,:,i)).^2))/sumi_hij_bg(i);
                sigma_next_bg(:,:,i) = diag(tmp_sigma);
                
                tmp_sigma = sum(hij_fg(:,i).*((TrainFG - mu_init_fg(:,:,i)).^2))/sumi_hij_fg(i);
                sigma_next_fg(:,:,i) = diag(tmp_sigma);
                
            end
            
            error_1_bg = mu_next_bg - mu_init_bg;
            error_2_bg = sigma_next_bg - sigma_init_bg;
            error_3_bg = pi_next_bg - pi_init_bg;
            
            error_1_fg = mu_next_fg - mu_init_fg;
            error_2_fg = sigma_next_fg - sigma_init_fg;
            error_3_fg = pi_next_fg - pi_init_fg;
            
            if all(error_1_bg < 10^-6,'all') && all(error_2_bg < 10^-6,'all') && all(error_1_bg < 10^-6,'all')...
                    && all(error_1_fg < 10^-6,'all') && all(error_2_fg < 10^-6,'all') && all(error_1_fg < 10^-6,'all')
                disp(iteration);
                break;
            end
            
            sigma_init_bg = sigma_next_bg;
            mu_init_bg = mu_next_bg;
            pi_init_bg = pi_next_bg;
            
            sigma_init_fg = sigma_next_fg;
            mu_init_fg = mu_next_fg;
            pi_init_fg = pi_next_fg;
        end
        
        mixture_sigma_FG(int2str(id1_mix)) = sigma_next_fg;
        mixture_mu_FG(int2str(id1_mix)) = mu_next_fg;
        mixture_pi_FG(int2str(id1_mix)) = pi_next_fg;
        mixture_sigma_BG(int2str(id1_mix)) = sigma_next_bg;
        mixture_mu_BG(int2str(id1_mix)) = mu_next_bg;
        mixture_pi_BG(int2str(id1_mix))= pi_next_bg;
        
    end
    
    %% Compute DCT
    
    zigZag = load("Zig-Zag Pattern.txt");
    zigZag = zigZag + 1;
    
    Cheetah = imread("cheetah.bmp");
    Cheetah = im2double(Cheetah);
    [x,y] = size(Cheetah);
    
    count = 1;
    
    Vectors64D = zeros((x-7)*(y-7), 64);
    Block_dct = zeros(1,64);
    for i=1:x-7
        for j=1:y-7
            Cheetah_block_dct = dct2(Cheetah(i:i+7,j:j+7));
            Block_dct(zigZag) = Cheetah_block_dct;
            Vectors64D(count,:) = Block_dct;
            count = count + 1;
        end
    end
    
    %% Read the cheetah mask
    
    desired = imread("cheetah_mask.bmp");
    desired = im2double(desired);
    desired_c = sum(desired,'all');
    desired_g = 255*270-desired_c;
    
    
    %% Prediction
    
    PoE_dim = zeros(num_mixtures*num_mixtures,size(list_dimensions,2));
    
    for id_dim = 1:size(list_dimensions,2)
        
        PoE = zeros(num_mixtures,num_mixtures);
        n_dimensions = list_dimensions(id_dim);
        
        for id1_mix = 1:num_mixtures
            
            % Cheetah figure features
            Image_result = zeros(255,270);
            
            mu_fg = mixture_mu_FG(int2str(id1_mix));
            sigma_fg = mixture_sigma_FG(int2str(id1_mix));
            pi_fg = mixture_pi_FG(int2str(id1_mix));
            
            for id2_mix = 1:num_mixtures
                
                % Cheetah figure features
                Image_result = zeros(255,270);
                
                mu_bg = mixture_mu_BG(int2str(id2_mix));
                sigma_bg = mixture_sigma_BG(int2str(id2_mix));
                pi_bg = mixture_pi_BG(int2str(id2_mix));
                
                Vectors_nD = Vectors64D(:,1:n_dimensions);
                
                %maha_dist = zeros((x-7)*(y-7),n_components);
                P_it_is_cheetah = zeros((x-7)*(y-7),n_components);
                P_it_is_grass = zeros((x-7)*(y-7),n_components);
                for id_comp = 1:n_components
                    
                    meanFG = mu_fg(:,1:n_dimensions,id_comp);
                    sigmaFG = sigma_fg(1:n_dimensions,1:n_dimensions,id_comp);
                    piFG = pi_fg(id_comp);
                    
                    meanBG = mu_bg(:,1:n_dimensions,id_comp);
                    sigmaBG = sigma_bg(1:n_dimensions,1:n_dimensions,id_comp);
                    piBG = pi_bg(id_comp);
                                        
                    P_it_is_cheetah(:,id_comp) =  multi_gaussian(Vectors_nD,meanFG,sigmaFG)*piFG;
                    P_it_is_grass(:,id_comp) =  multi_gaussian(Vectors_nD,meanBG,sigmaBG)*piBG;
                    
                end
                P_it_is_cheetah2 = sum(P_it_is_cheetah,2)*P_cheetah;
                P_it_is_grass2 = sum(P_it_is_grass,2)*P_grass;
                
                result_image = P_it_is_cheetah2 > P_it_is_grass2;
                Image_result(1:(x-7),1:(y-7)) = reshape(result_image,[(y-7),(x-7)])';
                
                error_matrix = Image_result-desired;
                error_c = sum(error_matrix==-1,'all');
                error_g = sum(error_matrix==1,'all');
                error = (error_c/desired_c)*P_cheetah + (error_g/desired_g)*P_grass;
                PoE(id1_mix,id2_mix) = error;
                
            end
        end
        
        PoE = PoE';
        PoE_dim(:,id_dim) = PoE(:);
        
    end
    
    plot(list_dimensions,PoE_dim,'DisplayName',[num2str(n_components),' components'],'LineWidth',2.0)
    hold on

end
legend('Location','northwest')

%% Plots

for plot_num = 0:num_mixtures*num_mixtures-1
    
    if mod(plot_num,4) == 0
        figure;        
        sgtitle(['Probability of Error vs. Dimension'])
    end
   
    subplot(2,2,mod(plot_num,4)+1)
    plot(list_dimensions,PoE_dim(plot_num+1,:))
    title(['Classifier',num2str(plot_num+1)])
    %title(['No. of components: ',num2str(n_components)])
    %ylim([0 0.2])
    xlabel('No. of dimensions')
    ylabel('PoE')
    
end

%% Function for Multi-variate Gaussian calculation

function [prob] = multi_gaussian(x,mu, sigma)

inv_sigma = inv(sigma);
var = diag(inv_sigma)';

power_nD = ((x - mu).^2).*var;
power = sum(power_nD,2);
dim = size(sigma,2);
prob = exp(-0.5*power)/(sqrt(((2*pi)^dim)*det(sigma)));
end
