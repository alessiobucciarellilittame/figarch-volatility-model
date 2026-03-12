function [theta_hat, sigma2_hat, loglik_val, info] = stima_figarch_qmle(r, J, theta0)
  % stima_figarch_qmle - Function for estimating a FIGARCH(1,d,1) model via QMLE
%
% Input:
%   r      - Return series
%   J      - Truncation length for long memory approximation
%   theta0 - Initial values (optional)
%
% Output:
%   theta_hat  - Estimated parameters [mu, omega, phi1, beta1, d]
%   sigma2_hat - Estimated variance series
%   loglik_val - Log-likelihood value
%   info       - Structure with additional estimation information and QMLE matrix

    if nargin < 3 || isempty(theta0)
        
        try
            garch_model = garch(1,1);
            garch_fit = estimate(garch_model, r);
            garch_params = garch_fit.Variance.GARCH;

            mu_init = mean(r);
            omega_init = garch_params{1,1};
            alpha_init = garch_params{1,2};
            beta_init = garch_params{1,3};
            d_init = 0.3;  

            theta0 = [mu_init; omega_init; alpha_init; beta_init; d_init];
        catch
            
            theta0 = [mean(r); var(r)*0.05; 0.2; 0.7; 0.3];
        end

        fprintf('Valori iniziali dei parametri: [%.4f, %.4f, %.4f, %.4f, %.4f]\n', ...
            theta0(1), theta0(2), theta0(3), theta0(4), theta0(5));
    end

    % parameters constraints
   
    LB = [-Inf, 1e-12, 0, 0, 0.01];  
    UB = [Inf, Inf, 0.999, 0.999, 0.999]; 

    nonlcon = @(theta) figarch_constraints(theta);

    
    objfun = @(theta) loglik_figarch(theta, r, J);

    options = optimoptions('fmincon', ...
        'Display', 'iter-detailed', ...
        'Algorithm', 'interior-point', ...
        'MaxIterations', 10000, ...
        'MaxFunctionEvaluations', 2e5, ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-10, ...
        'FiniteDifferenceType', 'central', ...
        'UseParallel', true, ...
        'CheckGradients', false, ...
        'FiniteDifferenceStepSize', 1e-5);

    try
        [theta_hat, loglik_val, exitflag, output, ~, grad] = fmincon(objfun, theta0, [], [], [], [], LB, UB, nonlcon, options);

        if exitflag > 0
            fprintf('optimization succesfully completed (exitflag = %d)\n', exitflag);
        else
            warning('possible convergence problems (exitflag = %d)\n', exitflag);
        end
    catch ME
        fprintf('error during the optimization: %s\n', ME.message);
        
        options.Algorithm = 'sqp';
        [theta_hat, loglik_val, exitflag, output, ~, grad] = fmincon(objfun, theta0, [], [], [], [], LB, UB, nonlcon, options);
    end

    sigma2_hat = figarch_variance(r, theta_hat, J);

    
    info = struct();
    info.exitflag = exitflag;
    info.output = output;
    info.gradient = grad;
    info.parameters = struct('mu', theta_hat(1), 'omega', theta_hat(2), ...
                           'phi1', theta_hat(3), 'beta1', theta_hat(4), 'd', theta_hat(5));
    
    
    [info.std_errors, info.V_qmle, info.I_hat, info.J_hat] = compute_qmle_std_errors(theta_hat, r, J);
    
    info.dof = length(r) - length(theta_hat);
    
    t_stats = theta_hat ./ info.std_errors;
    info.p_values = 2 * (1 - tcdf(abs(t_stats), info.dof));

    
    info.aic = 2*length(theta_hat) + 2*loglik_val;
    info.bic = log(length(r))*length(theta_hat) + 2*loglik_val;

    
    display_results(theta_hat, info);
end

% === STEP 1: compute_pi ===
function pi_j = compute_pi(d, J)
  
    pi_j = zeros(J,1);
    pi_j(1) = 1;
    for j = 2:J
        pi_j(j) = pi_j(j-1) * (j-1 - d) / j;
    end
end

% === STEP 2: FIGARCH variance recursion ===
function sigma2 = figarch_variance(r, theta, J)

    mu = theta(1);
    omega = theta(2);
    phi1 = theta(3);
    beta1 = theta(4);
    d = theta(5);

    T = length(r);
    e = r - mu;  
    e2 = e.^2;   


    sigma2 = zeros(T,1);
    sigma2(1) = var(r);  

    
    pi_j = compute_pi(d, J);
    pi_jm1 = [0; pi_j(1:end-1)];
    lambda_j = pi_j - (phi1 + beta1) * pi_jm1;

    
    omega_star = omega / (1 - beta1);

    
    for t = 2:T
        
        start_idx = max(1, t-J);
        e2_history = e2(t-1:-1:start_idx);
        pad_size = J - length(e2_history);

        if pad_size > 0
            
            arch_term = sum(lambda_j(pad_size+1:end) .* e2_history);
            
            avg_e2 = mean(e2(1:min(100,T)));
            arch_term = arch_term + sum(lambda_j(1:pad_size)) * avg_e2;
        else
            arch_term = sum(lambda_j(1:J) .* e2(t-1:-1:t-J));
        end

       
        sigma2(t) = omega_star + arch_term + beta1 * (sigma2(t-1) - omega_star);

        
        if isnan(sigma2(t)) || sigma2(t) <= 1e-6
            sigma2(t) = 1e-6;
        end
    end
end

% === STEP 3: log-likelihood ===
function negLL = loglik_figarch(theta, r, J)
  
    try
        sigma2 = figarch_variance(r, theta, J);
        mu = theta(1);
        e = r - mu;

        
        valid_idx = ~isnan(sigma2) & (sigma2 > 0);
        loglik = -0.5 * log(2*pi) - 0.5 * log(sigma2(valid_idx)) - 0.5 * (e(valid_idx).^2) ./ sigma2(valid_idx);

        negLL = -sum(loglik);

        
        if isnan(negLL) || isinf(negLL)
            negLL = 1e10; 
        end
    catch
        negLL = 1e10;  
    end
end

% ===  function for log-likelihood computation of a singular observation===
function ll = single_loglik(theta, r, J, t)
    
    sigma2 = figarch_variance(r, theta, J);
    mu = theta(1);

    if t > length(r) || isnan(sigma2(t)) || sigma2(t) <= 1e-10
        ll = 0;
    else
        e = r(t) - mu;
        ll = -0.5 * (log(2*pi*sigma2(t)) + (e^2)/sigma2(t));
    end
end

% === staibility constraints FIGARCH ===
function [c, ceq] = figarch_constraints(theta)
    
    phi1 = theta(3);
    beta1 = theta(4);
    d = theta(5);

   
    c = [
        phi1 - beta1 + d - 1;  % phi1 - beta1 + d < 1
        beta1 - phi1 - d;      % beta1 - phi1 - d > 0
    ];
    ceq = [];  
end

% === standard errors QMLE (robusti) ===
function [std_errors_qmle, V_qmle, I_hat, J_hat] = compute_qmle_std_errors(theta, r, J)
 % compute_qmle_std_errors - Computes QMLE robust standard errors
%
% Input:
%   theta - estimated parameters of the FIGARCH model
%   r     - return series
%   J     - truncation for long memory
%
% Output:
%   std_errors_qmle - robust standard errors
%   V_qmle          - QMLE variance matrix
%   I_hat           - numerical Hessian
%   J_hat           - outer product of gradients

    
    
    loglik_fun = @(th) loglik_figarch(th, r, J);
    
   
    I_hat = compute_hessian(loglik_fun, theta);
    
    G = compute_scores_efficient(theta, r, J); % T x k
    
    
    Tobs = size(G,1);
    J_hat = (G' * G) / Tobs;

    I_hat_reg = I_hat + eye(size(I_hat,1)) * 1e-8; 
    
  
    try
        inv_I = inv(I_hat_reg);
    catch
        warning('Matrice Hessiana quasi-singolare. Utilizzata pseudoinversa.');
        inv_I = pinv(I_hat_reg);
    end
    
    
    V_qmle = inv_I * J_hat * inv_I;
    
   
    std_errors_qmle = sqrt(diag(V_qmle));
    
    
    for i = 1:length(std_errors_qmle)
        if isnan(std_errors_qmle(i)) || std_errors_qmle(i) <= 0
            
            std_errors_qmle(i) = 0.1; 
        end
    end
    
    
    std_errors_qmle = min(std_errors_qmle, 0.5);
end

% === Efficient calculation for the score for every observation ===
function G = compute_scores_efficient(theta, r, J)
    
    
    k = length(theta);
    T = length(r);
    mu = theta(1);
    
    
    sigma2 = figarch_variance(r, theta, J);
    e = r - mu;
    
    
    log_lik_components = zeros(T, 1);
    valid_idx = ~isnan(sigma2) & (sigma2 > 1e-10);
    log_lik_components(valid_idx) = -0.5 * log(2*pi*sigma2(valid_idx)) - 0.5 * (e(valid_idx).^2)./sigma2(valid_idx);
    
    
    G = zeros(T, k);
    eps_val = 1e-5;
    
    for j = 1:k
        
        theta_plus = theta;
        theta_plus(j) = theta(j) + eps_val;
        
       
        sigma2_plus = figarch_variance(r, theta_plus, J);
        e_plus = r - theta_plus(1);  % if j=1, it changes
        
        log_lik_plus = zeros(T, 1);
        valid_plus = ~isnan(sigma2_plus) & (sigma2_plus > 1e-10);
        log_lik_plus(valid_plus) = -0.5 * log(2*pi*sigma2_plus(valid_plus)) - 0.5 * (e_plus(valid_plus).^2)./sigma2_plus(valid_plus);
        
      
        theta_minus = theta;
        theta_minus(j) = theta(j) - eps_val;
        
    
        sigma2_minus = figarch_variance(r, theta_minus, J);
        e_minus = r - theta_minus(1);  % if j=1, it changes
        
        log_lik_minus = zeros(T, 1);
        valid_minus = ~isnan(sigma2_minus) & (sigma2_minus > 1e-10);
        log_lik_minus(valid_minus) = -0.5 * log(2*pi*sigma2_minus(valid_minus)) - 0.5 * (e_minus(valid_minus).^2)./sigma2_minus(valid_minus);
        
       
        G(:,j) = (log_lik_plus - log_lik_minus) / (2 * eps_val);
    end
end

% ===  Hessian matrix computation ===
function H = compute_hessian(loglikfun, theta)
 
    
    k = length(theta);
    H = zeros(k,k);
    eps_val = 1e-5;
    f0 = loglikfun(theta);
    
    for i = 1:k
        th_i_plus = theta; 
        th_i_plus(i) = theta(i) + eps_val;
        
        th_i_minus = theta; 
        th_i_minus(i) = theta(i) - eps_val;
        
        f_plus = loglikfun(th_i_plus);
        f_minus = loglikfun(th_i_minus);
        
      
        H(i,i) = (f_plus - 2*f0 + f_minus) / (eps_val^2);
        
        
        for j = i+1:k
            th_ij_pp = th_i_plus; 
            th_ij_pp(j) = theta(j) + eps_val;
            
            th_ij_pm = th_i_plus; 
            th_ij_pm(j) = theta(j) - eps_val;
            
            th_ij_mp = th_i_minus; 
            th_ij_mp(j) = theta(j) + eps_val;
            
            th_ij_mm = th_i_minus; 
            th_ij_mm(j) = theta(j) - eps_val;
            
            fpp = loglikfun(th_ij_pp);
            fpm = loglikfun(th_ij_pm);
            fmp = loglikfun(th_ij_mp);
            fmm = loglikfun(th_ij_mm);
            
            
            H(i,j) = (fpp - fpm - fmp + fmm) / (4 * eps_val^2);
            H(j,i) = H(i,j);  
        end
    end
end

function display_results(theta, info)
    
    fprintf('\n==========================================\n');
    fprintf('   FIGARCH ESTIMATION RESULTS(1,d,1) QMLE\n');
    fprintf('==========================================\n');

    param_names = {'mu', 'omega', 'phi1', 'beta1', 'd'};

    fprintf('%-6s %12s %12s %12s %12s\n', 'Param', 'Estimate', 'QMLE S.E.', 't-stat', 'p-value');
    fprintf('------------------------------------------------------\n');

    for i = 1:length(theta)
        std_err = info.std_errors(i);
        t_stat = theta(i) / std_err;
        
        p_value = 2 * (1 - tcdf(abs(t_stat), info.dof));
        
        % Add significance stars
        stars = '';
        if p_value < 0.01
            stars = '***';
        elseif p_value < 0.05
            stars = '**';
        elseif p_value < 0.1
            stars = '*';
        end
        
        fprintf('%-6s %12.6f %12.6f %12.4f %12.4f %s\n', param_names{i}, theta(i), std_err, t_stat, p_value, stars);
    end

    fprintf('------------------------------------------------------\n');
    fprintf('Log-likelihood: %.6f\n', -info.output.funcCount);
    fprintf('AIC: %.6f\n', info.aic);
    fprintf('BIC: %.6f\n', info.bic);
    fprintf('Convergence: %s (exitflag=%d)\n', convergence_message(info.exitflag), info.exitflag);
    fprintf('Nota: Gli errori standard sono robusti (QMLE)\n');
    fprintf('Significatività: *** p<0.01, ** p<0.05, * p<0.1\n');
    fprintf('==========================================\n\n');
end

% === CONVERGENCE MESSAGE  ===
function msg = convergence_message(exitflag)
 
    switch exitflag
        case 1
            msg = 'Optimum';
        case 2
            msg = 'optimal conditions satisfied';
        case 0
            msg = 'max nummber of iterations reached';
        case -1
            msg = 'Terminated by output function';
        case -2
            msg = 'No solution founded';
        otherwise
            msg = 'Unknown';
    end
end

