
function metrics = compute_portfolio_metrics(r, rf)
    
    trading_days = 252;
    alpha = 0.01;
    mean_annual_ret = mean(r, 'omitnan') * trading_days;
    vol_annual = std(r, 'omitnan') * sqrt(trading_days);
    sharpe_ratio = (mean(r, 'omitnan') - rf) / std(r, 'omitnan') * sqrt(trading_days);
    neg_ret = r(r < 0);
    % neg_ret = r(r < rf);
    if ~isempty(neg_ret)
        sortino_ratio = (mean(r, 'omitnan') - rf) / std(neg_ret, 'omitnan') * sqrt(trading_days);
    else
        sortino_ratio = Inf; 
    end
    cum_curve = cumprod(1 + r);
    running_max = cummax(cum_curve);
    drawdown = (cum_curve - running_max) ./ running_max;
    max_drawdown = min(drawdown);
    
    % Calmar ratio
    calmar_ratio = mean_annual_ret / abs(max_drawdown);
    
    % Ulcer Index 
    ulcer_index = sqrt(mean(drawdown.^2, 'omitnan'));
    
    % Value at Risk (VaR)
    VaR = -quantile(r, alpha);
    
    % Return to VaR
    return_to_var = mean(r, 'omitnan') / VaR;
    
    % Output
    metrics = struct( ...
        'mean_annual_ret', mean_annual_ret, ...
        'vol_annual', vol_annual, ...
        'sharpe_ratio', sharpe_ratio, ...
        'sortino_ratio', sortino_ratio, ...
        'max_drawdown', max_drawdown, ...
        'calmar_ratio', calmar_ratio, ...
        'ulcer_index', ulcer_index, ...
        'VaR', VaR, ...
        'return_to_var', return_to_var ...
    );
end