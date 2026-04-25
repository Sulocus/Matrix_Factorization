% Demonstration of VAMP recovering K-sparse signals of dimension N from
% M = del*N noisy linear measurements, where K=rho*M.

addpath('../main')
addpath('../stateEvo')

% handle random seed
if verLessThan('matlab','7.14')
  defaultStream = RandStream.getDefaultStream;
else
  defaultStream = RandStream.getGlobalStream;
end
if 0 % new RANDOM trial
  savedState = defaultStream.State;
  save random_state.mat savedState;
else % repeat last trial
  load random_state.mat
end
defaultStream.State = savedState;

%% set simulation parameters
isCmplx = false; % use complex-valued signal, transform, and noise?
SNRdB = 40; % signal-to-noise ratio in dB [40]

% signal and measurement dimensions
L = 10; % number of signal vectors to recover [100]
N = 1024; % signal dimension [1024]
del = 0.5; % measurement rate M/N [0.5]
rho = 0.2; % normalized sparsity rate E{K}/M=(E{K}/N)/del [0.2]

% linear transform configuration, described through the svd A=U*S*V'
svType = 'cond_num'; % in {'cond_num','spread','low_rank'}
cond_num = 3; % condition number [2 is limit for AMP]
spread = 1.1; % amount to spread singular values (=1 means iid Gaussian A, =0 means frame) [1.1 is limit for AMP]
low_rank = round(min(N,round(del*N))/2);
UType = 'I'; % in {'DFT','DCT','DHT','DHTrice','Haar','I'}
VType = 'Haar'; % in {'DFT','DCT','DHT','DHTrice','Haar','I'}
shuffleV = true; % shuffle the rows of V' ?
randsignV = true; % randomly sign-flip the columns of V' ?
Afro2 = N; % squared Frobenius norm of matrix [N]

% plot and simulation control
runAMP = true; % run AMP for comparison?
runSE = true; % run state evolution?
runOracle = true; % calculate support oracle NMSE?
plot_traj = true; % plot NMSE trajectory of each column?
median_on = true; % report median instead of mean NMSE?
verbose = 0;

% other signal parameters
fixed_K = false; % used fixed sparsity E{K} versus random K?
xvar0 = 1; % prior variance of x coefs
xmean1 = 0; % prior mean of non-zero x coefs

%% set algorithm parameters
linearStage = 'lsqr'; % VAMP linear stage in {'exact','cg','lsqr','gd'} [lsqr]
solver_tol = 1e-5; % linear-stage solver tolerance (if not 'exact') [1e-5]
solver_iters = 1000; % linear-stage solver iterations (if not 'exact') [1000]
maxit = 100; % max iterations for VAMP
iters_admm = 0; % how many iterations of ADMM-PR to run prior to VAMP
tol = min(1e-3,max(1e-6,10^(-SNRdB/10))); % stopping tolerance for VAMP
damp = 1; % damping for VAMP, in (0,1]
denoiser = 'BG'; % in {'BG','DMM','MAPLaplace'}
learnPrior = false; % automatically tune the denoiser?
learnNoisePrec = false; % automatically tune the noise variance?

%% setup
M = round(del*N);
beta = rho*M/N; % probability of a non-zero coef
xvar1 = xvar0/beta; % prior variance of non-zero x coefs
wvar = (Afro2/M)*10^(-SNRdB/10)*beta*(abs(xmean1)^2+xvar1); 
if (strcmp(UType,'DFT')||strcmp(VType,'DFT'))&&(~isCmplx)
  warning('setting isCmplx=true since complex-valued matrix')
  isCmplx = true;
elseif isCmplx&&(~strcmp(UType,'DFT'))&&(~strcmp(VType,'DFT'))
  warning('setting isCmplx=false since real-valued matrix')
  isCmplx = false;
end

%% generate signal 
x = zeros(N,L);
for l=1:L
  if fixed_K
    supp = randperm(N,round(beta*N)); 
  else
    supp = find(rand(N,1)<beta); 
  end
  K = length(supp);
  if isCmplx
    x(supp,l) = xmean1 + sqrt(0.5*xvar1)*randn(K,2)*[1;1j];
  else
    x(supp,l) = xmean1 + sqrt(xvar1)*randn(K,1);
  end
end

% generate noise 
if isCmplx
  w = sqrt(0.5*wvar)*(randn(M,L) + 1j*randn(M,L));
else
  w = sqrt(wvar)*randn(M,L);
end

%% generate linear transform
switch svType
  case 'spread', svParam = spread;
  case 'cond_num', svParam = cond_num;
  case 'low_rank', svParam = low_rank;
end
mat = genMatSVD(M,N,UType,svType,svParam,VType,...
                'isCmplx',isCmplx,'Afro2',Afro2,...
                'shuffle',shuffleV,'randsign',randsignV,...
                'fxnHandles',true);
d = [mat.s.^2;zeros(M-length(mat.s),1)]; % need length(d)=M
A = mat.fxnA; Ah = mat.fxnAh;
U = mat.fxnU; Uh = mat.fxnUh;


% generate observation
z = A(x); 
SNRdB_test = 20*log10(norm(z(:))/norm(w(:)));
y = z + w;

%% compute support-oracle performance bound
if runOracle
  I = speye(N);
  x0 = zeros(N,L);
  oracleNMSEdB = nan(L,1);
  for l=1:L
    supp = find(x(:,l)~=0);
    try
      A0 = A(I(:,supp)); % fast but not compatible with all A(.)
    catch
      K = length(supp); % slow but compatible with all A(.)
      A0 = zeros(M,K);
      for k=1:K, A0(:,k) = A([zeros(supp(k)-1,1);1;zeros(N-supp(k),1)]); end
    end
    a0 = A0*ones(length(supp),1);
    x0(supp,l) = xmean1 + (A0'*A0+(wvar/xvar1)*eye(length(supp)))\(A0'*(y(:,l)-a0*xmean1)); 
    oracleNMSEdB(l) = 20*log10(norm(x0(:,l)-x(:,l))/norm(x(:,l)));
  end
end

%% establish denoiser
switch denoiser
case 'BG'
  if learnPrior
    betaInit = 1/N;
    xvar0init = xvar0;
    xvar1init = xvar0init/betaInit;
    tuneDim = 'col';
    if isCmplx
      if beta<1
        EstimIn = SparseScaEstim(CAwgnEstimIn(0,xvar1init,0,'autoTune',true,'tuneDim',tuneDim),betaInit,0,'autoTune',true,'tuneDim',tuneDim);
      elseif beta==1
        EstimIn = CAwgnEstimIn(0,xvar1init,0,'autoTune',true,'tuneDim',tuneDim);
      else
        error('invalid rho since rho>N/M')
      end
    else
      if beta<1
        EstimIn = SparseScaEstim(AwgnEstimIn(0,xvar1init,0,'autoTune',true,'tuneDim',tuneDim),betaInit,0,'autoTune',true,'tuneDim',tuneDim);
      elseif beta==1,
        EstimIn = AwgnEstimIn(0,xvar1init,0,'autoTune',true,'tuneDim',tuneDim);
      else
        error('invalid rho since rho>N/M')
      end
    end
  else
    if isCmplx
      EstimIn = SparseScaEstim(CAwgnEstimIn(xmean1,xvar1),beta);
    else
      EstimIn = SparseScaEstim(AwgnEstimIn(xmean1,xvar1),beta);
    end
  end
case 'DMM'
  alpha = 1.5; % controls thresholding
  debias = false;
  EstimIn = SoftThreshDMMEstimIn(alpha,'debias',debias);
  if learnPrior, 
    error('learnPrior not implemented for SoftThreshDMM'); 
  end;
case 'MAPLaplace'
  lam = 1/sqrt(wvar); % controls thresholding
  if learnPrior,
    EstimIn = SoftThreshEstimIn(lam,0,'autoTune',true,'counter',10) 
  else
    EstimIn = SoftThreshEstimIn(lam);
  end
otherwise
  error('unknown denoiser')
end

if 0
  display('temporary test to see if divergence is avoided!')
  clear EstimIn
  fxn_denoise = @(rhat,rvar) bsxfun(@times, ...
                          max(0,bsxfun(@minus,abs(rhat), ...
                          bsxfun(@times, alpha, sqrt(mean(rvar,1))) ...
                          )),sign(rhat));
  EstimIn = FxnhandleEstimIn(fxn_denoise);
end

%% setup VAMP
%gam1init = 1/(beta*(abs(xmean1).^2+xvar1) + wvar*sum(d)/sum(d.^2))
gam1init = sum(abs(y).^2,1)/sum(abs(A(randn(N,1))).^2,1);
vampOpt = VampSlmOpt3;
if strcmp(linearStage,'exact')
  vampOpt.U = U; vampOpt.Uh = Uh; vampOpt.d = d; 
else % linearStage is not 'exact'
  vampOpt.solver = linearStage; % in {'cg','lsqr','gd'}
  vampOpt.solver_tol = solver_tol;
  vampOpt.solver_iters = solver_iters;
  vampOpt.solver_silent = false;
  %vampOpt.divNum1=2; display('setting divNum1 > 1!')
end
vampOpt.Ah = Ah; 
vampOpt.N = N;
vampOpt.gam1init = gam1init;
vampOpt.damp = damp;
vampOpt.nitMax = maxit;
vampOpt.tol = tol;
vampOpt.iters_admm = iters_admm;
if learnNoisePrec
  vampOpt.learnNoisePrec = true;
  vampOpt.learnNoisePrecAlg = 'EM';
  vampOpt.learnNoisePrecNit = 100;
  vampOpt.learnNoisePrecTol = 0.01;
else
  vampOpt.learnNoisePrec = false;
  vampOpt.NoisePrecInit = 1/wvar; 
end
vampOpt.verbose = verbose;
vampOpt.fxnErr = @(x2) 10*log10( sum(abs(x2-x).^2,1)./sum(abs(x).^2,1) ); 


%% run VAMP
if strcmp(linearStage,'exact')
  [~,vampEstFin] = VampSlmEst3(EstimIn,y,A,vampOpt);
  %[~,vampEstFin] = VampSlmEst2(EstimIn,y,A,vampOpt);
else
  [~,vampEstFin] = VampSlmEst3_fastA(EstimIn,y,A,vampOpt);
  %vampLinsolveIters = vampEstFin.linsolve_iters
end
vampNMSEdB_ = vampEstFin.err; 
vampNit = vampEstFin.nit;

%% run VAMP state evolution
if runSE
  estInAvg = EstimInAvg(EstimIn,x);
  %gam1init = 1000 % to test replica, try starting from near-perfect initialization
  vampSeNMSE = VampSlmSE(estInAvg,d,N,wvar,vampNit,mean(gam1init),damp)./(beta*(abs(xmean1)^2+xvar1));
end
  
%% setup and run AMP
ampNit = 0;
if runAMP
  if learnPrior
    warning('AMP is using the prior learned by EM-VAMP')
  end
  Aamp = FxnhandleLinTrans(M,N,A,Ah,Afro2/(M*N));
  clear optAMP;
  optAMP = AmpOpt();
  tstart = tic;
  [~,optAMPfin,ampEstHist] = ampEst(EstimIn,y,Aamp,optAMP);
  time_amp = toc(tstart);
  ampNit = length(ampEstHist.it);
  ampNMSEdB_ = nan(L,ampNit);
  for l=1:L
    ampNMSEdB_(l,:) = 10*log10(sum(abs(ampEstHist.xhat((l-1)*N+[1:N],:)-x(:,l)*ones(1,ampNit)).^2,1)/norm(x(:,l))^2);
  end
  %figure(2); clf; gampShowHist(ampEstHist,optAMPfin,x); % debug AMP
end

%% plot NMSE trajectory for each of L recovered columns
vstr = denoiser; if learnPrior || learnNoisePrec, vstr = ['EM-',vstr]; end;
plotNit = max(vampNit,ampNit);
if plot_traj
  figure(1); clf;
  % plot VAMP
  handy = semilogx(1:vampNit,vampNMSEdB_,'b.-');
  set(handy(1),'Displayname',[vstr,'-VAMP']);
  % plot AMP
  if runAMP
    hold on;
      handy = [handy, semilogx(1:ampNit,ampNMSEdB_,'r.-')];
    hold off;
    set(handy(1,end),'Displayname',[vstr,'-AMP']);
  end
  % plot support oracle
  if runOracle
    ax = gca; ax.ColorOrderIndex = 1; % use same colors
    hold on; 
      handy = [handy, semilogx([1;plotNit],oracleNMSEdB*[1,1],'k--')]; 
    hold off;
    set(handy(1,end),'Displayname','oracle');
  else
    oracleNMSEdB = inf; % used below in axis 
  end
  % legend
  legend(handy(1,:)); 
  if median_on
    ylabel('median NMSE [dB]')
  else
    ylabel('average NMSE [dB]')
  end
  xlabel('iterations')
  grid on
  axis([1,plotNit,5*floor(min([vampNMSEdB_(:);oracleNMSEdB])/5),1])
end % plot_traj

%% plot average NMSE of VAMP
figure(2); clf;
if median_on
  vampNMSE_avg = median(10.^(vampNMSEdB_/10),1);
  oracleNMSE_avg = median(10.^(oracleNMSEdB/10),1); 
  if runAMP, ampNMSE_avg = median(10.^(ampNMSEdB_/10),1); end
else
  vampNMSE_avg = mean(10.^(vampNMSEdB_/10),1);
  oracleNMSE_avg = mean(10.^(oracleNMSEdB/10),1); 
  if runAMP, ampNMSE_avg = mean(10.^(ampNMSEdB_/10),1); end
end
plot(1:vampNit,vampNMSE_avg,'+-','Displayname',[vstr,'-VAMP']);
set(gca,'YScale','log','XScale','log')
grid on
xlabel('iteration')
if median_on
  ylabel('median NMSE [dB]')
else
  ylabel('average NMSE [dB]')
end

%% plot state-evolution, AMP, and support oracle
hold on;
  if runSE
    semilogx(1:vampNit,vampSeNMSE,'o-','Displayname','VAMP SE'); 
  end
  if runAMP
    semilogx(1:ampNit,ampNMSE_avg,'x-','Displayname',[vstr,'-AMP']); 
  end
  if runOracle
    semilogx([1,plotNit],[1,1]*oracleNMSE_avg,'-.','Displayname','oracle');
  end
hold off;
legend(gca,'show')
axis([1,plotNit,10^floor(log10(min([vampNMSE_avg,oracleNMSE_avg,vampSeNMSE]))),1])

%% plot gammas
if 0
    figure(3); clf;
    subplot(211)
        loglog([1:vampNit],vampEstHist.gam1)
        grid on
        ylabel('gam1')
    subplot(212)
        loglog([1:vampNit],vampEstHist.gam2)
        grid on
        ylabel('gam2')
    figure(4); clf;
        semilogx([1:vampNit],vampEstHist.gam1./vampEstHist.gam2)
        grid on
        ylabel('gam1/gam2')
end