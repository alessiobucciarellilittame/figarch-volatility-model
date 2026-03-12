function [portfolio_returns_all, weights_all, risk_metrics] = run_multi_risk_figarch_portfolio(returns, window_size, fallback_window, max_iter, optimization_type, custom_options)
% RUN_MULTI_RISK_FIGARCH_PORTFOLIO Optimizes portfolio weights using FIGARCH volatility models and various VaR measures
%
% SYNTAX:
%   [portfolio_returns_all, weights_all, risk_metrics] = run_multi_risk_figarch_portfolio(returns, window_size, fallback_window, max_iter, optimization_type, custom_options)
%
% INPUTS:
%   returns - Table or matrix of returns with assets in columns and time in rows
%   window_size - Rolling window size for model estimation (default: 250)
%   fallback_window - Window size for fallback volatility estimation (default: 30)
%   max_iter - Maximum number of iterations for FIGARCH estimation (default: 6000)
%   optimization_type - Risk measure used to optimize the portfolio:
%      'tstudent' - VaR using Student's t-distribution
%      'gaussian' - VaR using Gaussian distribution
%      'cornish-fisher' - VaR using Cornish-Fisher expansion
%      'historical-var' - VaR based on historical data
%      'all' - Runs all VaR-based methods and returns results in a multi-column format
%   custom_options - Structure with optional custom parameters:
%      .alpha - Confidence level for VaR (default: 0.99)
%      .nu - Degrees of freedom for t-distribution (default: estimated from data)
%      .figarchOrder - Order parameters for FIGARCH model [p,q,d] (default: [1,1,0.4])
%      .lambda - Trade-off parameter between risk and return (default: 0.1)
%      .cov_method - Method for covariance estimation ('diagonal','dcc','constant') (default: 'diagonal')
%      .rebalancing_frequency - Portfolio rebalancing frequency (default: 1)
%
% OUTPUTS:
%   portfolio_returns_all - Table with portfolio returns for each optimization method
%   weights_all - Table with portfolio weights for each asset over time
%   risk_metrics - Structure with risk and performance metrics for each method
%
% EXAMPLES:
%   [returns, weights] = run_multi_risk_figarch_portfolio(asset_returns, 250, 30, 5000, 'all');


   
    if nargin < 2 || isempty(window_size), window_size = 250; end
    if nargin < 3 || isempty(fallback_window), fallback_window = 30; end
    if nargin < 4 || isempty(max_iter), max_iter = 6000; end
    if nargin < 5 || isempty(optimization_type), optimization_type = 'tstudent'; end
    if nargin < 6, custom_options = struct(); end
    
   
    options = struct();
    options.alpha = 0.99;                         % confidence level for VaR
    options.nu = [];                              % degrees of freedom
    options.figarchOrder = [1, 1, 0.4];           % orders FIGARCH model [p,q,d]
    options.lambda = 0.1;                         % parameter trade-off risk-return
    options.cov_method = 'diagonal';              % covariance estimation method
    options.rebalancing_frequency = 1;            % rebalance every N periods
    options.min_sample_size = 50;                 % miminum sample dimenson
    options.differentiate_methods = true;         % it onliges methods to use different approaches
    options.J = 100;                              %  truncation parameter for FIGARCH
    
    
    fn = fieldnames(custom_options);
    for i = 1:length(fn)
        options.(fn{i}) = custom_options.(fn{i});
    end
    
    
    fprintf('starting portfolio optimization using FIGARCH model and VaR measures\n'); 
    fprintf('Optimization method: %s\n', optimization_type);
    fprintf('Window dimension : %d, fallback window: %d\n', window_size, fallback_window);
    fprintf('Confidence level (alpha): %.4f\n', options.alpha);
    
     
    if ~istable(returns)
        warning("Input is not a table, it is automatically converted.");
        returns = array2table(returns, 'VariableNames', strcat("Asset", string(1:size(returns,2))));
    end
    

    dates = returns.Properties.RowNames;
    if isempty(dates)
        dates = string((1:height(returns))');
    else
        dates = string(dates);
    end
    
    
    numeric_vars = varfun(@isnumeric, returns, 'OutputFormat', 'uniform');
    strategies = returns.Properties.VariableNames(numeric_vars);
    returns_data = returns(:, strategies);
    [T, N] = size(returns_data);
    
   
    run_all = strcmp(optimization_type, 'all');
    if run_all
        methods = {'tstudent', 'gaussian', 'cornish-fisher', 'historical-var'};
        num_methods = length(methods);
    else
        methods = {optimization_type};
        num_methods = 1;
    end
    
    
    portfolio_returns_matrix = nan(T - window_size, num_methods);
    weights_tensor = nan(T - window_size, N, num_methods);
    
   
    if any(strcmp(methods, 'tstudent')) && isempty(options.nu)
      
        try
            options.nu = estimate_t_distribution_dof(table2array(returns_data));
            fprintf('degrees of freedom estimated for t distribution: %.2f\n', options.nu);
        catch
            options.nu = 6;  % Default fallback
            fprintf('use of default degrees of freedom: %.2f\n', options.nu);
        end
    end
    
    
    t_quantile = tinv(options.alpha, options.nu);
    normal_quantile = norminv(options.alpha);
    
 
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool('local', min(feature('numcores'), 8));
    end
    

    window_data_cache = cell(T-window_size+1, 1);
    for t = window_size:T
        window_data_cache{t-window_size+1} = table2array(returns_data(t-window_size+1:t, :));
    end
    
    
    fprintf('Optimization advance:     ');
    for t_idx = 1:(T-window_size)
      
        if mod(t_idx, max(1, floor((T-window_size)/20))) == 0
            fprintf('\b\b\b\b%3d%%', round(100*t_idx/(T-window_size)));
        end
        
        t = window_size + t_idx - 1;
        window_data = window_data_cache{t_idx};
        
        
        if mod(t_idx-1, options.rebalancing_frequency) ~= 0 && t_idx > 1
         
            for m = 1:num_methods
                weights_tensor(t_idx, :, m) = weights_tensor(t_idx-1, :, m);
                if t+1 <= T
                    next_returns = table2array(returns_data(t+1, :))';
                    portfolio_returns_matrix(t_idx, m) = weights_tensor(t_idx, :, m) * next_returns;
                end
            end
            continue;
        end
        
        
        vol_estimates = nan(N, 1);
        mu_estimates = nan(N, 1);
        skew_estimates = nan(N, 1);
        kurt_estimates = nan(N, 1);
        figarch_diagnostics = cell(N, 1);
        
        parfor i = 1:N
            serie = window_data(:, i);
            serie = serie(~isnan(serie));
            
            try
                
                if length(serie) < options.min_sample_size
                    error('too short series for FIGARCH estimation');
                end
                
                
                [parameters, sigma2_hat, residuals, info] = stima_figarch_semplificato(...
                    serie, options.J, max_iter, options.figarchOrder);
                
                if info.exitflag > 0
                    vol_estimates(i) = sqrt(sigma2_hat(end));
                    mu_estimates(i) = mean(serie);
                    
                    
                    if any(strcmp(methods, 'cornish-fisher'))
                        skew_estimates(i) = skewness(residuals);
                        kurt_estimates(i) = kurtosis(residuals) - 3; 
                    end
                    
                
                    figarch_diagnostics{i} = struct(...
                        'parameters', parameters, ...
                        'convergence', info.exitflag, ...
                        'iterations', info.iterations, ...
                        'final_vol', sqrt(sigma2_hat(end)), ...
                        'log_likelihood', info.fval);
                else
                    error('FIGARCH estimation do not converge');
                end
            catch err
                
                fallback_slice = serie(max(1, end-fallback_window+1):end);
                vol_estimates(i) = std(fallback_slice, 'omitnan');
                mu_estimates(i) = mean(fallback_slice, 'omitnan');
                
                if any(strcmp(methods, 'cornish-fisher'))
                    skew_estimates(i) = skewness(fallback_slice);
                    kurt_estimates(i) = kurtosis(fallback_slice) - 3;
                end
                
                figarch_diagnostics{i} = struct('error', err.message, 'fallback', true);
            end
        end
        
      
        valid_idx = ~isnan(vol_estimates);
        if sum(valid_idx) < 2
            warning('Not enough valid assets at time %d, jump', t);
            continue;
        end
        
        sigma_vec = vol_estimates(valid_idx);
        mu_vec = mu_estimates(valid_idx);
        skew_vec = skew_estimates(valid_idx);
        kurt_vec = kurt_estimates(valid_idx);
        
        
        n_valid = sum(valid_idx);
        
        switch options.cov_method
            case 'diagonal'
                
                Sigma = diag(sigma_vec.^2);
                
            case 'dcc'
                
                try
                    valid_data = window_data(:, valid_idx);
                    [~, H] = estimate_dcc_garch(valid_data);
                    Sigma = H(:,:,end); 
                catch
                  
                    Sigma = diag(sigma_vec.^2);
                end
                
            case 'constant'
             
                valid_data = window_data(:, valid_idx);
                corr_mat = corrcoef(valid_data);
                D = diag(sigma_vec);
                Sigma = D * corr_mat * D;
                
            otherwise
                
                Sigma = diag(sigma_vec.^2);
        end
        
        
        [V, D] = eig(Sigma);
        D = diag(max(diag(D), 1e-6));
        Sigma = V * D * V';
        
       
        if t+1 <= T
            next_returns = table2array(returns_data(t+1, valid_idx))';
        else
            next_returns = [];
        end
        
        
        if ~isempty(next_returns) && any(isnan(next_returns))
            continue;
        end
        
      
        for m = 1:num_methods
            current_method = methods{m};
            
            
            switch current_method
                case 'tstudent'
                  
                    risk_func = @(w) -w'*mu_vec + t_quantile * sqrt(w' * Sigma * w * (options.nu - 2) / options.nu);
                    
                case 'gaussian'
                
                    risk_func = @(w) -w'*mu_vec + normal_quantile * sqrt(w' * Sigma * w);
                    
                case 'cornish-fisher'
              
                    risk_func = @(w) improved_cornish_fisher_var(w, mu_vec, Sigma, skew_vec, kurt_vec, options.alpha);
                    
                case 'historical-var'
                    
                    risk_func = @(w) calculate_historical_var(w, window_data(:, valid_idx), options.alpha) - options.lambda * w' * mu_vec;
                    
                otherwise
                    error('Metodo di ottimizzazione non supportato: %s', current_method);
            end
            
            
            w0 = ones(n_valid, 1) / n_valid;
            Aeq = ones(1, n_valid);
            beq = 1;
            lb = zeros(n_valid, 1);
            
       
            opt_settings = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
                'MaxIterations', 1000, 'TolFun', 1e-8, 'TolX', 1e-8);
          
            try
                w_opt = fmincon(risk_func, w0, [], [], Aeq, beq, lb, [], [], opt_settings);
            catch
                
                w_opt = w0;
                warning('Optimization failed for method %s at time %d, use of uniform weights', current_method, t);
            end
           
            if ~isempty(next_returns)
                portfolio_returns_matrix(t_idx, m) = w_opt' * next_returns;
            end
            
            
            full_weights = zeros(1, N);
            full_weights(valid_idx) = w_opt';
            weights_tensor(t_idx, :, m) = full_weights;
        end
    end
    fprintf('\b\b\b\b100%%\n');
    
    
    for m = 1:num_methods
        portfolio_returns_matrix(:, m) = interpola_valori_mancanti(portfolio_returns_matrix(:, m));
        for i = 1:N
            weights_tensor(:, i, m) = interpola_valori_mancanti(weights_tensor(:, i, m));
        end
    end
    
   
    final_dates = dates((window_size+1):T);
    
  
    if run_all
        portfolio_returns_all = array2table(portfolio_returns_matrix, 'VariableNames', methods);
    else
        portfolio_returns_all = array2table(portfolio_returns_matrix, 'VariableNames', {optimization_type});
    end
    portfolio_returns_all.Properties.RowNames = final_dates;
    
 
    weights_all = cell(num_methods, 1);
    for m = 1:num_methods
        weights_all{m} = array2table(weights_tensor(:, :, m), 'VariableNames', strategies);
        weights_all{m}.Properties.RowNames = final_dates;
    end
    

    if num_methods == 1
        weights_all = weights_all{1};
    end
    
   
    risk_metrics = calculate_performance_metrics(portfolio_returns_matrix, methods, options.alpha);
    
  
    fprintf('\nPerformances sum up:\n');
    fprintf('%-15s  %10s  %10s  %10s  %10s  %10s\n', 'method', 'Rend.Ann.', 'Vol.Ann.', 'Sharpe', 'MaxDD', 'CVaR 99%');
    fprintf('----------------------------------------------------------------------\n');
    
    for m = 1:num_methods
        fprintf('%-15s  %10.2f%%  %10.2f%%  %10.2f  %10.2f%%  %10.2f%%\n', ...
            methods{m}, ...
            risk_metrics.AnnualReturn(m) * 100, ...
            risk_metrics.AnnualVol(m) * 100, ...
            risk_metrics.Sharpe(m), ...
            risk_metrics.MaxDrawdown(m) * 100, ...
            risk_metrics.CVaR99(m) * 100);
    end
 
end
%% support function

function [dof] = estimate_t_distribution_dof(returns)
    
    % degrees of freedom estimation for t-distribution from return data
    
    
    standardized_returns = returns ./ std(returns, 0, 1, 'omitnan');
    
    all_returns = standardized_returns(:);
    all_returns = all_returns(~isnan(all_returns));
    
    t_nll = @(nu) -sum(log(tpdf(all_returns, nu)));
    
    options = optimset('Display', 'off');
    [dof, ~, exitflag] = fminbnd(t_nll, 2.1, 30, options);
    
    if exitflag <= 0
        
        dof = 6;
    end
end

function [var_cf] = improved_cornish_fisher_var(w, mu, Sigma, skew, kurt, alpha)
    

    port_mu = w' * mu;
    port_vol = sqrt(w' * Sigma * w);
    
  
    port_skew = 0;
    port_kurt = 0;
    
    n = length(w);
    for i = 1:n
     
        port_skew = port_skew + w(i)^3 * skew(i) * (sqrt(Sigma(i,i))^3) / (port_vol^3);
        
        
        port_kurt = port_kurt + w(i)^4 * kurt(i) * (Sigma(i,i)^2) / (port_vol^4);
       
        for j = 1:n
            if i ~= j
                port_kurt = port_kurt + w(i)^2 * w(j)^2 * (Sigma(i,j)^2) / (port_vol^4);
            end
        end
    end
    

    z_alpha = norminv(alpha);
    

    h1 = (z_alpha^2 - 1) / 6;
    h2 = (z_alpha^3 - 3*z_alpha) / 24;
    h3 = -(2*z_alpha^3 - 5*z_alpha) / 36;
    
    cf_quantile = z_alpha + h1*port_skew + h2*port_kurt + h3*port_skew^2;
    
   
    var_cf = -port_mu + port_vol * cf_quantile;
end

function var_hist = calculate_historical_var(w, historical_returns, alpha)
    
    

    if isempty(historical_returns) || size(historical_returns, 1) < 10
        var_hist = Inf;
        return;
    end
    
 
    portfolio_returns = historical_returns * w;
    
    
    sorted_returns = sort(portfolio_returns);
    index = max(1, ceil(size(historical_returns, 1) * (1-alpha)));
    
    
    var_hist = -sorted_returns(index);
end

% function vol = calculate_ewma_volatility(returns, lambda)
%     % Calcola volatilità EWMA (Exponentially Weighted Moving Average)
% 
%     if isempty(returns) || length(returns) < 5
%         vol = std(returns, 'omitnan');
%         return;
%     end
% 
%     returns = returns(~isnan(returns));
%     n = length(returns);
% 
%     % Inizializza con varianza campionaria iniziale
%     var_t = var(returns(1:min(5, n)));
% 
%     % Calcola varianza EWMA
%     for t = 2:n
%         var_t = lambda * var_t + (1 - lambda) * returns(t-1)^2;
%     end
% 
%     vol = sqrt(var_t);
% end

function [params, H] = estimate_dcc_garch(data)
    
    [T, n] = size(data);
    

    h = zeros(T, n);
    residuals = zeros(T, n);
    
    for i = 1:n
    
        try
            garch_spec = garch(1, 1);
            [~, ~, ~, garch_params] = estimate(garch_spec, data(:,i), 'Display', 'off');
            [h(:,i), residuals(:,i)] = simulate_garch11(data(:,i), garch_params.GARCH{1}, garch_params.ARCH{1}, garch_params.Constant);
        catch
           
            rolling_window = 20;
            h(1:rolling_window,i) = var(data(1:rolling_window,i));
            for t = rolling_window+1:T
                h(t,i) = var(data(t-rolling_window+1:t,i));
            end
            residuals(:,i) = data(:,i) ./ sqrt(h(:,i));
        end
    end
    

    std_resid = residuals ./ sqrt(h);
    
    
    R_bar = corrcoef(std_resid, 'rows', 'complete');
    

    a = 0.05;
    b = 0.90;
    
   
    Q_bar = R_bar;
    Q = zeros(n, n, T);
    R = zeros(n, n, T);
    H = zeros(n, n, T);
    
  
    Q(:,:,1) = Q_bar;
    R(:,:,1) = Q_bar ./ sqrt(diag(Q_bar) * diag(Q_bar)');
    
   
    for t = 2:T
        e = std_resid(t-1,:)';
        Q(:,:,t) = (1-a-b)*Q_bar + a*(e*e') + b*Q(:,:,t-1);
        Q_diag = diag(diag(Q(:,:,t)).^(-0.5));
        R(:,:,t) = Q_diag * Q(:,:,t) * Q_diag;
      
        D = diag(sqrt(h(t,:)));
        H(:,:,t) = D * R(:,:,t) * D;
    end
    
    
    params = struct('a', a, 'b', b, 'R_bar', R_bar);
end

function [h, resid] = simulate_garch11(returns, beta, alpha, omega)
   
    T = length(returns);
    h = zeros(T, 1);
    resid = zeros(T, 1);
    

    h(1) = var(returns(1:min(20,T)));
    resid(1) = returns(1);
    

    for t = 2:T
        h(t) = omega + alpha * resid(t-1)^2 + beta * h(t-1);
        resid(t) = returns(t);
    end
end

function metrics = calculate_performance_metrics(returns, methods, alpha)
    
    
    num_methods = length(methods);
    metrics = struct();
    

    trading_days = 252;
    

    metrics.AnnualReturn = zeros(num_methods, 1);
    metrics.AnnualVol = zeros(num_methods, 1);
    metrics.Sharpe = zeros(num_methods, 1);
    metrics.MaxDrawdown = zeros(num_methods, 1);
    metrics.CVaR99 = zeros(num_methods, 1);
    metrics.Return_CVaR_Ratio = zeros(num_methods, 1);
    metrics.VaR99 = zeros(num_methods, 1);
    metrics.ExpectedShortfall = zeros(num_methods, 1);
    
    for m = 1:num_methods
        rets = returns(:, m);
        rets = rets(~isnan(rets));
        
        if ~isempty(rets)
         
            metrics.AnnualReturn(m) = prod(1 + rets)^(trading_days/length(rets)) - 1;
            
     
            metrics.AnnualVol(m) = std(rets) * sqrt(trading_days);
            
       
            metrics.Sharpe(m) = metrics.AnnualReturn(m) / metrics.AnnualVol(m);
            
           
            cumulative = cumprod(1 + rets);
            running_max = zeros(size(cumulative));
            running_max(1) = cumulative(1);
            for i = 2:length(cumulative)
                running_max(i) = max(running_max(i-1), cumulative(i));
            end
            drawdowns = (cumulative ./ running_max) - 1;
            metrics.MaxDrawdown(m) = min(drawdowns);
            
    
            sorted_returns = sort(rets);
            cutoff_index = ceil((1-alpha) * length(sorted_returns));
            metrics.VaR99(m) = -sorted_returns(cutoff_index);
            
  
            metrics.CVaR99(m) = mean(sorted_returns(1:cutoff_index));
            metrics.ExpectedShortfall(m) = metrics.CVaR99(m);
            

            metrics.Return_CVaR_Ratio(m) = metrics.AnnualReturn(m) / abs(metrics.CVaR99(m));
        end
    end
end

function interpolated = interpola_valori_mancanti(data)


    if all(isnan(data))
        interpolated = data;
        return;
    end
    

    idx = 1:length(data);
    valid_idx = find(~isnan(data));
    
    if length(valid_idx) < 2
       
        interpolated = fillmissing(data, 'nearest');
    else
 
        interpolated = interp1(valid_idx, data(valid_idx), idx, 'linear', 'extrap');
    end
end