%% Demonstration of noise variance estimation using expectation maximization
%Method described in "An Expectation-Maximization Approach to Tuning
%Generalized Vector Approximate Message Passing"
close all;
clear;clc;

nitEM=2;
EMtype_VAMP=2;%Noise variance estimation method. Use 2 for the high SNR approximation

%%Problem Settings
%Set up the signal
n = 16^2; % signal dimension [16^2]
p = round(8*n); % number of phaseless measurements
k = round(0.999*n); % signal sparsity
x_o = zeros(n,1); x_o(randperm(n,k)) = randn(k,1)+1i*randn(k,1);
xvar_nz = var(x_o(x_o~=0)); % variance of nonzero coefs
xvar = var(x_o); % variance of all coefs
sigma_w = sqrt(20);
noise_ests=[-3:.5:1];%.5:1];
matrix_type = 3;

%Create variables to store results
K=length(noise_ests);
R=5;%Number of trials to run
prVAMP_fixed_NSME_array=zeros(K,R);
prVAMP_adaptive_NSME_array=zeros(K,R);
prVAMP_adaptive_final_wvar_hat=zeros(K,R);
for r=1:R
    %Generate signal
    x_o =(randn(n,1)+1i*randn(n,1));
    xvar_nz=sqrt(2);
    xvar=sqrt(2);
    xvar_nz_init = xvar_nz; % iniital nonzero-coef variance (set >> xvar_nz !)
    
    %Simulate measurements
    switch matrix_type
        case 1
            A = round(rand(p,n));% 0,1 Measurements
        case 2
            A = 2*round(rand(p,n))-1;% -1,1 Measurements
        case 3
            A = 1/sqrt(2)*(randn(p,n)+1i*randn(p,n));% Gaussian Measurements
        case 4
            A = randn(p,n);% Gaussian Measurements
    end
    noise_vec = sigma_w*(1/sqrt(2)*randn(p,1)+1i*1/sqrt(2)*randn(p,1));
    z_o = A*x_o;
    y = abs(z_o+noise_vec);
    
    spars_init = 0.999; % initial sparsity rate (set near 1 !)
    tuneDelayVamp = 25; % number of iterations to wait before tuning%Should not be used

for k=1:K
    noise_ests(k)
    wvar_init=10^(noise_ests(k))*sigma_w^2;

    %Set up the initialization
    xvar_init = xvar;
    x_init = sqrt(xvar_init)*(1/sqrt(2)*randn(n,1)+1i*1/sqrt(2)*randn(n,1)); 

    %%Simulation Settings
    plot_on = false; % plot NMSE versus iteration? (slows things down!)

    %Set up prVAMP
    [U,S,V]=svd(A,'econ');
    d=diag(S).^2;
    vampOpt = VampGlmOpt;
    vampOpt.nitMax = 2e2;
    vampOpt.tol = 1e-6;
    vampOpt.damp = 0.8; % try 0.8; 1 means no damping
    vampOpt.dampGam = 0.5; % try 0.5; 1 means no damping
    %vampOpt.dampConfig = [0,1,0,1,0,0, 1,0,0,0,1,0]; % original from VampGlmEst
    %vampOpt.dampConfig = [0,1,1,1,0,0, 0,0,0,1,0,1]; % best from dampTest
    vampOpt.dampConfig = [0,1,1,1,0,0, 0,0,0,0,0,1]; % best from dampTest
    vampOpt.verbose = false;
    vampOpt.U = U;
    vampOpt.V = V;
    vampOpt.d = d;
    vampOpt.r1init = x_init;
    vampOpt.gam1xinit = 1/xvar_init;
    vampOpt.p1init = A*x_init;%y_ind;
    vampOpt.gam1zinit = 1/var(y);
    vampOpt.silent = true;
    vampOpt.altUpdate = false;
    
    wvar_hat_vamp = [wvar_init,nan(1,nitEM+1)];

    
    %%prVAMP
    % Reset prior since internal parameters changed while running prGAMP
    EstimIn = SparseScaEstim(...
           CAwgnEstimIn(0,xvar_nz_init,false,...
                        'autoTune',false,...
                        'mean0Tune',false,...
                        'counter',tuneDelayVamp),...
           spars_init,false,'autoTune',false,'counter',tuneDelayVamp);
    
    %Run prVAMP
    clear vampHist;
    t0=cputime;
    % loop over EM noise-learning iterations
    for em=1:nitEM+1

%         % use externally estimated noise variance
        if em==1
            EstimOut = ncCAwgnEstimOut(y,wvar_hat_vamp(em)*ones(p,1),false,false);
        else
            vampOpt = optFin.warmStart(vampFin,'tol',vampOpt.tol);
            EstimOut = ncCAwgnEstimOut(y,wvar_hat_vamp(em)*ones(p,1),false,false);
        end

          [x_prVAMP,vampFin,optFin,vampHist] = VampGlmEst2(EstimIn,EstimOut,A,vampOpt); % slow
        if em==1
            x_prVAMP_init=x_prVAMP;
        end
        switch EMtype_VAMP
            case 0
                error('VAMP does not support variance estimation with EstimOut');
            case 1
                wvar_hat_vamp(em+1) = mean(2*(y-abs(A*gampFin.xhat)).^2);
            case 2 %High SNR Approximation
                complex_intgrl=zeros(p,1);
                for i=1:p
                    etak=vampFin.eta2z;
                    zk_i=vampFin.z2(i);
                    L_arg=-etak*abs(zk_i)^2;
                    expI0=besseli(0,-L_arg/2,1);
                    expI1=besseli(1,-L_arg/2,1);
                    L_half=(1-L_arg)*expI0-L_arg*expI1;
                    E_rho_i=sqrt(pi/(4*etak))*L_half;
                    E_rho2_i=1/etak+abs(zk_i)^2;
                    complex_intgrl(i)=y(i)^2-2*y(i)*E_rho_i+E_rho2_i;
                end
                wvar=sigma_w^2;
                wvar_hat_vamp(em+1) = 2/p*sum(complex_intgrl);
            otherwise
                error('Unknown EMtype')
        end   
    end
    t_prVAMP = cputime-t0;
    x_prVAMP = x_prVAMP(:);
    prVAMP_NMSE_init = norm(x_o-disambig1Dfft(x_prVAMP_init,x_o))^2/norm(x_o)^2;
    prVAMP_NMSE = norm(x_o-disambig1Dfft(x_prVAMP,x_o))^2/norm(x_o)^2;

    
    % Print results
    display(['prVAMP Reconstruction NMSE=',num2str(prVAMP_NMSE),', time=',num2str(t_prVAMP),', sparsity_est=',num2str(EstimIn.p1)]);

    
%     prGAMP_NMSE_array(k,r)=prGAMP_NMSE;
    prVAMP_fixed_NSME_array(k,r)=prVAMP_NMSE_init;
    prVAMP_adaptive_NSME_array(k,r)=prVAMP_NMSE;
    prVAMP_adaptive_final_wvar_hat(k,r)=wvar_hat_vamp(end);
end
end

figure(1);
subplot(1,2,1);
stem(10.^(noise_ests(:))*sigma_w^2,[median(prVAMP_fixed_NSME_array,2),median(prVAMP_adaptive_NSME_array,2)]);
set(gca,'xscal','log')
legend({'Fixed','EM'},'FontSize',16);
xlim([10.^(noise_ests(1))*sigma_w^2,10.^(noise_ests(end))*sigma_w^2])
ylim([0,.2]);
set(gca, 'FontName', 'Times')
set(gca, 'FontSize',14);
xlabel('\sigma_w^2 Initialization','FontSize',18);
ylabel('NMSE','FontSize',18);
subplot(1,2,2);
h1=stem(10.^(noise_ests(:))*sigma_w^2,median(prVAMP_adaptive_final_wvar_hat,2),'r');
hold on;
h2=plot(10.^(noise_ests(:))*sigma_w^2,wvar*ones(K,1),'b');
legend([h1],{'\sigma_w^2 Estimate'},'FontSize',16);
set(gca,'xscal','log')
set(gca, 'FontName', 'Times')
set(gca, 'FontSize',14);
xlabel('\sigma_w^2 Initialization','FontSize',18);
ylabel('\sigma_w^2 Estimate','FontSize',18);
xlim([10.^(noise_ests(1))*sigma_w^2,10.^(noise_ests(end))*sigma_w^2])
ylim([0,200]);
h=gcf;
set(h,'pos',[100 100 800 300])
