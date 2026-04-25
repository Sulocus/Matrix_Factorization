% this script tests main/TuneVarScaleEstimIn.m and main/TuneVarEstimIn.m

rng(1)

% signal parameters
xType = 'bg'; % in bpsk,qpsk,bg,cbg
beta = [1,0.5]; % r = beta*x + N(0,rvar)
%beta = [exp(1j*2*pi*0.1)/2,0.5]; % r = beta*x + N(0,rvar)
rvar = [0.1,0.1].*beta.^2; % r = beta*x + N(0,rvar)

% denoiser parameters
estType = 3; % 1=EstimIn, 2=TuneVarEstimIn, 3=TuneVarScaleEstimIn
tuneDim = 'col'; % in 'joint','col','row'
nit = 1000; % EM iterations

% establish prior 
assert(all(size(beta)==size(rvar)),'beta must have same size as rvar')
switch xType
  case 'bpsk'
    if length(beta)~=1 
      warning('using beta(1) since DisScaEstim only works on column vectors!')
      beta = beta(1); rvar = rvar(1);
    end
    estIn0 = DisScaEstim([1;-1], [0.5;0.5]);
    isCmplx = false;
  case 'qpsk'
    if length(beta)~=1 
      warning('using beta(1) since DisCScaEstim only works on column vectors!')
      beta = beta(1); rvar = rvar(1);
    end
    estIn0 = DisCScaEstim([1;-1;1j;-1j], [0.25;0.25;0.25;0.25]);
    isCmplx = true;
  case 'bg' 
    rate_nz = 0.1;
    mean_nz = 0;
    var_nz = (1-mean_nz^2)/rate_nz; assert(var_nz>0); % E{x(n)^2}=1
    estIn0 = SparseScaEstim(AwgnEstimIn(mean_nz,var_nz),rate_nz);
    isCmplx = false;
  case 'cbg' 
    rate_nz = 0.1;
    mean_nz = 0;
    var_nz = (1-mean_nz^2)/rate_nz; assert(var_nz>0); % E{|x(n)|^2}=1
    estIn0 = SparseScaEstim(CAwgnEstimIn(mean_nz,var_nz),rate_nz);
    isCmplx = true;
  otherwise
    error('unrecognized xType');
end

% handle complex beta
if any(~isreal(beta)), isCmplx = true; end

% wrap with TuneVar
switch estType
  case 1
    estIn = estIn0;
    titStr = 'EstimIn';
  case 2
    estIn = TuneVarEstimIn(estIn0,'tuneDim',tuneDim,'nit',nit);
    titStr = 'TuneVarEstimIn';
  case 3
    estIn = TuneVarScaleEstimIn(estIn0,'tuneDim',tuneDim,'nit',nit);
    titStr = 'TuneVarScaleEstimIn';
  otherwise
    error('unrecognized estType')
end

% generate signal and noisy measurement
switch tuneDim
  case 'row'
    N = length(rvar); L = 10000; % use few rows and many columns
    x = zeros(N,L);
    rhat = zeros(N,L);
    for n=1:N
      x(n,:) = estIn.genRand(L);
      if isCmplx
        rhat(n,:) = beta(n)*x(n,:) + sqrt(rvar(n)/2)*[1;1j]*randn(2,L);
      else
        rhat(n,:) = beta(n)*x(n,:) + sqrt(rvar(n))*randn(1,L);
      end
    end
  otherwise
    N = 10000; L = length(rvar); % use few columns and many rows
    x = zeros(N,L);
    rhat = zeros(N,L);
    for l=1:L
      x(:,l) = estIn.genRand(N);
      if isCmplx
        rhat(:,l) = beta(l)*x(:,l) + sqrt(rvar(l)/2)*randn(N,2)*[1;1j];
      else
        rhat(:,l) = beta(l)*x(:,l) + sqrt(rvar(l))*randn(N,1);
      end
    end
end

% set initial denoiser variance
switch tuneDim
  case 'joint'
    rvar_init = mean(rvar)*ones(N,L);
  case 'col'
    rvar_init = ones(N,1)*rvar;
  case 'row'
    rvar_init = rvar'*ones(1,L);
  otherwise
    error('unrecognized tuneDim')
end
rvar_init = 10*rvar_init;

% denoise measurements
[xhat,xvar] = estIn.estim(rhat,rvar_init);

% extract beta and rvar trajectories 
switch estType
  case 1 % EstimIn
    switch tuneDim
      case 'joint'
        beta_hat = ones(nit+1,1);
        rvar_hat = ones(nit+1,1)*rvar_init(1);
      case 'col'
        beta_hat = ones(nit+1,L);
        rvar_hat = ones(nit+1,1)*rvar_init(1,:);
      case 'row'
        beta_hat = ones(nit+1,N);
        rvar_hat = ones(nit+1,N)*rvar_init(:,1);
    end
  case 2 % TuneVarEstimIn
    switch tuneDim
      case 'joint'
        beta_hat = ones(nit+1,1);
      case 'col'
        beta_hat = ones(nit+1,L);
      case 'row'
        beta_hat = ones(nit+1,N);
    end
    rvar_hat = estIn.rvarHist;
  case 3 % TuneVarScaleEstimIn
    beta_hat = estIn.scaleHist;
    rvar_hat = estIn.rvarHist;
end

% plot final estimates 
figure(1); clf;
N1 = min(N*L,50);
if isCmplx
  subplot(211)
    plot(1:N1,real(x(1:N1)),'+', 1:N1,real(xhat(1:N1)),'o')
    legend('x','xhat','Location','Best')
    ylabel('real')
    title(titStr)
    grid on;
  subplot(212)
    plot(1:N1,imag(x(1:N1)),'+', 1:N1,imag(xhat(1:N1)),'o')
    legend('x','xhat','Location','Best')
    ylabel('imag')
    xlabel('index')
    grid on;
else
  plot(1:N1,x(1:N1),'+', 1:N1,xhat(1:N1),'o')
  legend('x','xhat','Location','Best')
  xlabel('index')
  ylabel('estimate')
  title(titStr)
  grid on;
end

% print error in dB
switch tuneDim
  case 'joint'
    nmseXdB_autotune = 20*log10(norm(xhat-x,'fro')/norm(x,'fro'))
  case 'col'
    nmseXdB_autotune = 20*log10(sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1))
  case 'row'
    nmseXdB_autotune = 20*log10(sum(abs(xhat-x).^2,2)./sum(abs(x).^2,2)).'
end

% plot trajectories of beta and rvar estimates
figure(2); clf;
if isCmplx
  subplot(311)
    handy = plot([0:nit]'*ones(1,size(beta_hat,2)),real(beta_hat),'.-');
    set(handy,'DisplayName','estimated');
    hold on;
      handy = plot([0:nit]'*ones(1,length(beta)),ones(nit+1,1)*real(beta),'--');
      set(handy,'DisplayName','true');
    hold off;
    legend('Location','Best')
    ylabel('real(beta)')
    title(titStr)
    grid on;
  subplot(312)
    handy = plot([0:nit]'*ones(1,size(beta_hat,2)),imag(beta_hat),'.-');
    set(handy,'DisplayName','estimated');
    hold on;
      handy = plot([0:nit]'*ones(1,length(beta)),ones(nit+1,1)*imag(beta),'--');
      set(handy,'DisplayName','true');
    hold off;
    legend('Location','Best')
    ylabel('imag(beta)')
    grid on;
  subplot(313)
    handy = semilogy([0:nit]'*ones(1,size(beta_hat,2)),rvar_hat,'.-');
    set(handy,'DisplayName','estimated');
    hold on;
      handy = semilogy([0:nit]'*ones(1,length(rvar)),ones(nit+1,1)*rvar,'--');
      set(handy,'DisplayName','true');
    hold off;
    legend('Location','Best')
    ylabel('rvar')
    xlabel('EM iteration')
    grid on;
else
  subplot(211)
    handy = plot([0:nit]'*ones(1,size(beta_hat,2)),beta_hat,'.-');
    set(handy,'DisplayName','estimated');
    hold on;
      handy = plot([0:nit]'*ones(1,length(beta)),ones(nit+1,1)*beta,'--');
      set(handy,'DisplayName','true');
    hold off;
    legend('Location','Best')
    ylabel('beta')
    title(titStr)
    grid on;
  subplot(212)
    handy = semilogy([0:nit]'*ones(1,size(beta_hat,2)),rvar_hat,'.-');
    set(handy,'DisplayName','estimated');
    hold on;
      handy = semilogy([0:nit]'*ones(1,length(rvar)),ones(nit+1,1)*rvar,'--');
      set(handy,'DisplayName','true');
    hold off;
    legend('Location','Best')
    ylabel('rvar')
    xlabel('EM iteration')
    grid on;
end

% test to make sure that xvar is consistent with divergence
if strcmp(tuneDim,'col') | (length(beta)==1) 
  denoiser = @(rhat,rvar) estIn.estim(rhat,rvar);
  estIn2 = FxnhandleEstimIn(denoiser); 
  [~,xvar2] = estIn2.estim(rhat,rvar_init); % monte-carlo xvar estimate
  xvar_autotune = mean(xvar,1) 
  xvar_montecarlo = mean(xvar2,1)
end
