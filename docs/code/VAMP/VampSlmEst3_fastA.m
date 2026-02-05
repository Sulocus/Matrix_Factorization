function [x2,estFin,estHist] = VampSlmEst3_fastA(denoiser,y,A,opt)
%
% VampSlmEst3_fastA:  VAMP for the Standard Linear Model with fast "A"
%
% DESCRIPTION
% -----------
% This implements VAMP for the standard linear model, 
% i.e., estimating a vector x from measurements 
%    y = A*x + w ,
% where w is additive white Gaussian noise and A is a known linear operator.
%
% VAMP-SLM is an iterative algorithm where each iteration has two stages.
% The first stage performs MMSE (linear) estimation of x under the prior 
% p(x)=N(x;r1,1/gam1) and the likelihood p(y|x)=N(y;A*x,1/gamwHat), 
% where gamwHat is (an estimate of) the noise precision.  This stage 
% yields an estimate x1 with precision eta1.  The pair (x1,eta1) is then 
% converted into the pair (r2,gam2), where r2 is treated as an 
% AWGN-corrupted version of the true x with noise precision gam2.  
% The second stage of the iteration denoises r2 to produce the estimate 
% x2 with precision eta2.  Finally, the pair (x2,eta2) is converted to 
% the pair (r1,gam1), concluding the iteration.  
% These iterations are repeated until r1 converges, and the final 
% value of x2 reported as the estimate of x.
%
%
% USAGE
% -----
% [x2,estFin,estHist] = VampSlmEst3_fastA(denoiser,y,A,[opt])
%
% denoiser: estimates x from r = x + noise.
% Several options:
%   1) handle to fxn of form [xhat,xvar] = denoiser(r,rvar), where r is
%      the noisy signal and rvar is the noise variance
%   2) handle to fxn of form xhat = denoiser(r,rvar)
%   3) EstimIn object from GAMPmatlab
%
% y: an Mx1 vector of measurements (or MxL matrix in the MMV case) 
%
% A: linear operator.  Several options:
%   1) explicit matrix of size MxN
%   2) fxn handle, in which case opt.N must specify the # of columns in A
%      and opt.Ah must be a fxn handle to the adjoint operator
%   3) LinTrans object from GAMPmatlab
%
% opt: options structure, left empty or initialized via opt=VampSlmOpt3.
%   See VampSlmOpt3.m for details...
%
%
% IMPORTANT NOTE ON FAST OPERATORS
% --------------------------------
% For the linear estimation stage, VampSlmEst3_fastA solves a linear
% system of equations at each iteration using one of the following:
% a) conjugate gradient,
% b) LSQR,
% c) gradient descent with optimal stepsize,
% as selected in the VampSlmOpt3 options structure.
% The advantage is that "A" can be a high-dimensional fast operator, e.g., 
% based on, e.g., FFTs or fast wavelet transforms.  If your "A" is an explicit 
% matrix, or if you have a (fast) function handle for "U" in the eigendecomp
% "[U,D]=eig(A*A')", then it may be better to use VampSlmOpt3 instead of 
% VampSlmOpt3_fastA.


% Process output format
if (nargout > 3)
    error('too many output arguments')
end
saveHist = (nargout >= 3);

% Get options
if (nargin < 3) || isempty(opt)
    opt = VampSlmOpt3();  % default options
end
nitMax = opt.nitMax;  % maximum number of iterations
tol = opt.tol;  % stopping tolerance
gamMin = opt.gamMin;  % minimum allowed precision
gamMax = opt.gamMax;  % maximum allowed precision
damp = opt.damp;  % damping parameter
damp_auto = opt.damp_auto;  % auto damping?
learnNoisePrec = opt.learnNoisePrec; % learn noise precision?
learnNoisePrecAlg = opt.learnNoisePrecAlg; % learning algorithm
learnNoisePrecNit = opt.learnNoisePrecNit; % nit for noise learning
learnNoisePrecTol = opt.learnNoisePrecTol; % tol for noise learning
gamwHatMax = opt.NoisePrecMax; % max allowed noise precision
verbose = opt.verbose; % verbose output? 
silent = opt.silent; % disable all warnings?
iters_admm = opt.iters_admm; % how many iterations of ADMM-PR before VAMP
T1 = opt.divNum1; % number of perturbations for alf1 divergence 
T2 = opt.divNum2; % number of perturbations for alf2 divergence
dampDiv1 = opt.divDamp1;  % damping parameter for lin-stage divergences
dampDiv2 = opt.divDamp2;  % damping parameter for denoiser divergences

% Process measurements y
if isa(y,'numeric')
    [M,L] = size(y);
    if (~silent) && all(y(:)==zeros(M*L,1))
      warning('y is zero valued!  May cause problems...')
    end
else
    error('2nd input (y) must be a vector (or matrix) of measurements')
end

% Process linear transform A, length N
if isa(A,'numeric')

    N = size(A, 2); 
    fxnA = @(q) A*q;
    fxnAh = @(q) A'*q; 

elseif isa(A,'function_handle') || any(strcmp(superclasses(A),'LinTrans'))

    % treat the case where A is a function handle or a LinTrans object
    if isa(A,'function_handle')

        fxnA = A;
        if isa(opt.Ah,'function_handle')
            fxnAh = opt.Ah;
        else
            error('opt.Ah must be a fxn handle to a linear operator');
        end
        if ~isempty(opt.N)
            if floor(opt.N)==opt.N
                N = opt.N;
            else
                error('opt.N must be an integer');
            end
        else
            error('Since 3rd argument is a fxn handle, must specify opt.N');
        end

    else % A is a LinTrans object

        fxnA = @(x) A.mult(x);
        fxnAh = @(x) A.multTr(x);
        [~,N] = size(A);

    end

    % test fxnA for size
    try
        z = fxnA(zeros(N,L));
        if any(size(z)~=[M,L])
            error('output size of 3rd input (A) doesnt match 2nd input (y)');
        end
        clear z;
    catch
        error('3rd input (A) doesnt accept inputs of correct size')
    end

    % test fxnAh for size
    try
        x = fxnAh(zeros(M,L));
        if any(size(x)~=[N,L])
            error('output size of opt.Ah doesnt match 2nd input (y)');
        end
        clear x;
    catch
        error('opt.Ah doesnt accept inputs of correct size')
    end

else
    error('3rd input must be a matrix or fxn handle to a linear operator')

end

% Process denoiser
if isa(denoiser,'function_handle')
    if nargin(denoiser)~=2, error('need nargin(denoiser)==2'); end
    % check if this function handle returns [xhat,xvar] or just [xhat]
    try % to produce two outputs
        [~,~] = denoiser(zeros(N,L),ones(1,L));
        fxnDenoise = @(rhat,rvar) denoiser(rhat,rvar); 
        fxnDenoise_novar = @(rhat, rvar) first_output(denoiser, rhat, rvar); 
    catch % else turn into an EstimIn object
        denoiser1 = FxnhandleEstimIn(denoiser,...
                                     'changeFactor',opt.divChange,...
                                     'avg',opt.divNum2,'divMin',1e-5); 
        fxnDenoise = @(rhat,rvar) denoiser1.estim(rhat,rvar); 
        fxnDenoise_novar = denoiser; % no monte-carlo divergence!
    end 
elseif any(strcmp(superclasses(denoiser),'EstimIn')) 
    % turns EstimIn object into a function handle
    fxnDenoise = @(rhat,rvar) denoiser.estim(rhat,rvar); 
    fxnDenoise_novar = fxnDenoise;
else
    error(['First input (denoiser) must be either an EstimIn object, ',...
           'a fxn handle that accepts [rhat,rvar] and produces [xhat,xvar], ',...
           'or a fxn handle that accepts [rhat,rvar] and produces only xhat.']);
end 

% Process error-reporting function
if isa(opt.fxnErr,'function_handle')
    fxnErr = opt.fxnErr;
else
    fxnErr = [];
end

% Process general reporting function
if isa(opt.fxnGen,'function_handle')
    fxnGen = opt.fxnGen;
else
    fxnGen = [];
end

% Process stop function
if isa(opt.fxnStop,'function_handle')
    fxnStop = opt.fxnStop;
else
    fxnStop = [];
end

% Initialize noise precision & learning
gamwHatMin = M*L/norm(y,'fro')^2;
if ~isempty(opt.NoisePrecInit)
    gamwHat = opt.NoisePrecInit;
    if gamwHat>gamwHatMax
        warning('opt.NoisePrecInit too large')
        gamwHat = gamwHatMax;
    elseif gamwHat<gamwHatMin
        warning('opt.NoisePrecInit too small')
        gamwHat = gamwHatMin;
    end
else
    gamwHat = gamwHatMin;
end

% Prepare for saving history
if saveHist
    histIntvl = opt.histIntvl;
    nitSave = floor(nitMax/histIntvl);
    estHist.r1 = nan(N,L,nitSave);
    estHist.gam1 = nan(L,nitSave);
    estHist.x1 = nan(N,L,nitSave);
    estHist.gamwHat = nan(1,nitSave);
    estHist.eta1 = nan(L,nitSave);
    estHist.r2 = nan(N,L,nitSave);
    estHist.gam2 = nan(L,nitSave);
    estHist.x2 = nan(N,L,nitSave);
    estHist.eta2 = nan(L,nitSave);
    estHist.err = nan(L,nitSave);
    estHist.gen = nan(L,nitSave);
end

% Initialize r variables
if ~isempty(opt.r1init)
    r1 = opt.r1init;
    if size(r1,2)==1, r1 = r1*ones(1,L); end
else
    r1 = zeros(N,L); % default initialization
end
r1old = inf*ones(N,L); % used by fxnGen and fxnStop
r2old = inf*ones(N,L); % used by fxnGen and fxnStop

% Initialize gam1 precisions
if ~isempty(opt.gam1init)
    gam1 = opt.gam1init;
else
    gam1 = sum(abs(y).^2,1)/sum(abs(A(randn(N,1))).^2,1);
end
if size(gam1,2)==1, gam1 = gam1*ones(1,L); end

% Initialization specific to ADMM
if iters_admm > 0
    % initialize gam2 precision
    if ~isempty(opt.gam2init) 
        gam2 = opt.gam2init;
        if size(gam2,2)==1, gam2 = gam2*ones(1,L); end
    else
        gam2 = gam1;
    end
    eta1 = gam1 + gam2; 
    eta2 = eta1; 
    alf1 = gam1./eta1; 
    alf2 = gam2./eta2; 
end

% Initializatios of linear-system solver and divergence computation
state_x = zeros(1,L); % warm-start for x1 solver
v1 = cell(1,T1); 
for t=1:T1
    v1{t}=zeros(1,L); % warm-start for alf1 divergence computation
end
s1 = sign(randn(N,L,T1)); % perturbations for alf1 divergence 


% Set damping
damp1 = damp; % use same damping for R1 and gam1
damp2 = 1; % use same damping for R2 and gam2
damp_min = 0.05;

% ---------------------
% Run VAMP
err = nan(L,nitMax);
gen = nan(L,nitMax);
time = nan(1,nitMax);
gamwHat_ = nan(1,nitMax);
nitNoise = nan(1,nitMax);
tstart = tic;
i = 0;
stop = false;
flag = 0; 
linsolve_iters = nan(T1+1,nitMax); % iters for est and up to T1 div tries
%denoiser_calls = nan(1, nitMax); 

% Main loop of algorithm
while ~stop

    %-----update counter
    i = i + 1;

    if i <= iters_admm
        update_precisions = false;
    else
        update_precisions = true;
    end
    % disp(update_precisions);

    %-----damp r1 and gam1
    if (i > 1) && update_precisions 
        if damp_auto
          dampA = 2*min(alf1, alf2); % auto damping
          %dampA = 2*min(alf1,1-alf1);  % auto damping
          %dampA = 2*min(alf2,1-alf2);  % auto damping
          dampA1 = max(damp_min,min(dampA,damp1));
        else
          dampA1 = damp1;
        end
        r1 = bsxfun(@times,dampA1,r1) + bsxfun(@times,1-dampA1,r1old);
        gam1 = 1./(dampA1./sqrt(gam1) + (1-dampA1)./sqrt(gam1old)).^2;
    end
       
    %-----linear stage: estimation
    % x1 = argmin_x {0.5||A*x-y||^2 + 0.5*gam1/gamwHat||x-r1||^2}
    %    = inv(A'*A + (gam1/gamwHat)*eye(N))*(A'*y + (gam1/gamwHat)*r1)
    [x1,it,state_x] = LinSolve(y,fxnA,fxnAh,r1,gam1/gamwHat,state_x,opt); 
    linsolve_iters(1,i) = it; % record # linsolve iters 

    %-----linear stage: divergence computation
    % alf1 = (gam1/gamwHat)*tr(inv(A'*A + (gam1/gamwHat)*eye(N)))/N
    %     ~= (gam1/gamwHat)*s1'*inv(A'*A+(gam1/gamwHat)*eye(N))*s1/N 
    %        with random {+1,-1} s1
    if i>1, alf1old = alf1; end
    linsolve_it_div = zeros(T1,1); % # linsolve iters for div computation
    if update_precisions
        ll = [1:L]; % column indices at which to (re)compute divergence
        LL = length(ll); % number of columns to (re)compute
        maxTry = 10; % how many attempts to compute a good divergence 
        tries = 0; 
        alf1 = zeros(1,L); 
        while (tries < maxTry) && (LL > 0)
            alf1_trial = zeros(1,LL,T1); 
            for t=1:T1 % only compute divergence for necessary columns
                [out,it,state_div] = LinSolve(zeros(M,LL),fxnA,fxnAh,s1(:,ll,t),...
                                    gam1(ll)/gamwHat,v1{t}(:,ll),opt); 
                alf1_trial(:,:,t) = real(mean(s1(:,ll,t).*out,1)); 
                linsolve_it_div(t) = linsolve_it_div(t) + it; % record # iters
                if all(ll==[1:L])
                    v1{t} = state_div; % save for future warm start
                else
                    v1{t}(:,ll) = state_div; % save for future warm start
                end
            end
            alf1(:,ll) = mean(alf1_trial,3); 
            ll = find((alf1<0)|(alf1>1)); % exceeds limits...try again 
            LL = length(ll);
            tries = tries + 1;
            if (LL > 0)  
              s1(:,ll,:) = sign(randn(N,LL,T1)); % draw new perturbations
            end
        end 
        alf1 = max(min(alf1,1),0); % enforce limits
        
        % apply damping to alpha
        if i>1
            alf1 = (dampDiv1*sqrt(alf1) + (1-dampDiv1)*sqrt(alf1old)).^2;
        end
        eta1 = gam1./alf1; % reported but not used
    end
    linsolve_iters(2:end,i) = linsolve_it_div; % iterations for div tries
    
    % Debug: Exactly compute linear stage for comparison
    if 0
        if ~isa(A,'numeric')  
          A = fxnA(eye(N)); % create explicit matrix ... may take time
        end
        alf1_exact = zeros(1,L);
        x1_exact = zeros(N,L);
        for l=1:L; % chose one index
            Cinv_l = pinv(A'*A+(gam1(l)/gamwHat)*eye(N));
            x1_exact(:,l) = Cinv_l*(A'*y(:,l) + (gam1(l)/gamwHat)*r1(:,l));
            alf1_exact(l) = (gam1(l)/gamwHat/N)*trace(Cinv_l);
        end % for l
        err_x1 = sum(abs(x1-x1_exact).^2,1)./sum(abs(x1_exact).^2,1)
        alf1_exact
        alf1
        linsolve_iters_i = linsolve_iters(:,i)'
        
        if 0
            % Use exact x1 and alf1 in place of approximate ones
            x1 = x1_exact;
            alf1 = alf1_exact;
            eta1 = gam1./alf1; % reported but not used
            display('using exact x1 and alf1')
        end
    end
       
    %-----learn noise precision via EM
    if learnNoisePrec
        error('learnNoisePrec is not yet implemented')
    end 
    gamwHat_(i) = gamwHat;
    
    %-----update r2 and gam2
    if i>1, r2old = r2; gam2old = gam2; end
    r2 = bsxfun(@rdivide,(x1-bsxfun(@times,alf1,r1)),1-alf1);
    if update_precisions
        gam2 = (1-alf1)./alf1.*gam1;
        if (~silent)&&any(gam2<gamMin)
            warning('gam2 too small, iter=%i',i);
        end
        if (~silent)&&any(gam2>gamMax)
            warning('gam2 too large, iter=%i',i);
        end
        gam2 = min(max(gam2,gamMin),gamMax);
    end

    %-----damp r2 and gam2
    if (i > 1) && update_precisions 
        r2 = bsxfun(@times,damp2,r2) + bsxfun(@times,1-damp2,r2old);
        gam2 = 1./(damp2./sqrt(gam2) + (1-damp2)./sqrt(gam2old)).^2;
    end
    
    %-----second half of iteration
    if i>1, alf2old = alf2; end
    if update_precisions
        % compute denoiser output and variance 
        [x2,x2var] = fxnDenoise(r2, 1./gam2);
        alf2 = mean(x2var,1).*gam2; 

        if i>1 
            alf2 = (dampDiv2*sqrt(alf2) + (1-dampDiv2)*sqrt(alf2old)).^2;
        end
        eta2 = gam2./alf2; % reported but not used
    else
        % compute denoiser output but not variance (to save computation)
        x2 = fxnDenoise_novar(r2, 1./gam2);
    end

    %-----record progress
    if ~isempty(fxnErr)
        err(:,i) = fxnErr(x2).'; % evaluate error function
    end
    if ~isempty(fxnGen)
        gen(:,i) = fxnGen(r1old,r1,gam1,x1,eta1,...
                          r2old,r2,gam2,x2,eta2); % evaluate general function
    end
    
    %-----save history
    if saveHist && rem(i,histIntvl)==0
        iHist = i/histIntvl;
        estHist.r1(:,:,iHist) = r1;
        estHist.gam1(:,iHist) = gam1;
        estHist.x1(:,:,iHist) = x1;
        estHist.gamwHat(iHist) = gamwHat;
        estHist.eta1(:,iHist) = eta1;
        estHist.r2(:,:,iHist) = r2;
        estHist.gam2(:,iHist) = gam2;
        estHist.x2(:,:,iHist) = x2;
        estHist.eta2(:,iHist) = eta2.';
        estHist.err(:,iHist) = err(:,i).';
        estHist.gen(:,iHist) = gen(:,i).';
    end
    
    %-----second half of iteration (continued)
    r1old = r1;
    r1 = bsxfun(@rdivide,(x2-bsxfun(@times,alf2,r2)),1-alf2);
    gam1old = gam1;
    if update_precisions
        gam1 = (1-alf2)./alf2.*gam2;
        gam1 = min(max(gam1,gamMin),gamMax);
    end

    if (~silent)&&any(gam1<gamMin)
        warning('gam1 too small, iter=%i',i);
    end
    if (~silent)&&any(gam1>gamMax)
        warning('gam1 too large, iter=%i',i);
    end
    
    %-----report progress
    if verbose && rem(i,histIntvl)==0
        if isscalar(dampR1), dampR1 = dampR1*ones(1,L); end
        if ~isempty(fxnErr)
              fprintf(['i=%3i: ',...
                  'dampR1=%5.3f ',...
                  'gam1=%8.3g ',...
                  'alf1=%7.4f ',...
                  'gam2=%8.3f ',...
                  'alf2=%7.4f ',...
                  'eta2/eta1=%6.2f ',...
                  'err=%8.3g\n'],...
                  [i*ones(L,1),...
                  dampR1.',...
                  gam1.',...
                  alf1.', ...
                  gam2.',...
                  alf2.',...
                  (eta2./eta1).',...
                  err(:,i)].');
        end
    end
    
    %-----stopping rule
    if ~isempty(fxnStop)
        stop = fxnStop(i,err(:,i),r1old,r1,gam1,x1,eta1,...
            r2old,r2,gam2,x2,eta2);
        if stop 
            flag = 1; 
        end
    end
    if all(sqrt(sum(abs(r1-r1old).^2)./sum(abs(r1).^2)) < tol)||(i>=nitMax)
        stop = true;
    end
    
    %-----measure time until iteration i
    time(i) = toc(tstart);
end % while

% Export Outputs
estFin.r1old = r1old;
estFin.r1 = r1;
estFin.gam1 = gam1;
estFin.x1 = x1;
estFin.gamwHat = gamwHat_(1:i);
estFin.nitNoise = nitNoise(1:i);
estFin.eta1 = eta1;
estFin.r2 = r2;
estFin.gam2 = gam2;
estFin.x2 = x2; % this is the main output: the final estimate of x
estFin.eta2 = eta2;
estFin.nit = i;
estFin.err = err(:,1:i);
estFin.gen = gen(:,1:i);
estFin.time = time(1:i);
estFin.linsolve_iters = linsolve_iters(:,1:i);
%estFin.denoiser_calls = denoiser_calls(1:i); 
estFin.fxnStopFlag = flag; 

% Trim history
if saveHist
    iTrim = 1:floor(i/histIntvl);
    estHist.r1 = squeeze(estHist.r1(:,:,iTrim));
    estHist.gam1 = estHist.gam1(:,iTrim);
    estHist.x1 = squeeze(estHist.x1(:,:,iTrim));
    estHist.gamwHat = estHist.gamwHat(:,iTrim);
    estHist.eta1 = estHist.eta1(:,iTrim);
    estHist.r2 = squeeze(estHist.r2(:,:,iTrim));
    estHist.gam2 = estHist.gam2(:,iTrim);
    estHist.x2 = squeeze(estHist.x2(:,:,iTrim));
    estHist.eta2 = estHist.eta2(:,iTrim);
    estHist.err = estHist.err(:,iTrim);
end

end % VampSlmWhEst



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = first_output(f, r, rvar)
    [x, ~] = f(r, rvar); 
end


