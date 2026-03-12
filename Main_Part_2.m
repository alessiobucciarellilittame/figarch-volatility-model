clearvars  
clc        
close all  
%%
T = readtable('HFRX2.xlsx');
T = flipud(T);
DataOnly = T(:, 2:end); 
DataMatrix = DataOnly.Variables;
LogReturns = log(DataMatrix(2:end, :) ./ DataMatrix(1:end-1, :));
LogReturns = [NaN(1, size(LogReturns, 2)); LogReturns];
T_log = array2table(LogReturns, 'VariableNames', DataOnly.Properties.VariableNames);
T_log = addvars(T_log, T.Name, 'Before', 1, 'NewVariableNames', 'Name');

%%
T_log.Properties.RowNames = cellstr(string(T_log.Name));
T_log.Name = []; 

%% Parameters and benchmark separation
window_size = 250;
fallback_window = 30;
max_iter = 6000;
custom_options = struct('alpha', 0.99, 'lambda', 0.1, 'cov_method', 'diagonal','rebalancing_frequency',1);
benchmark_name = 'Benchmark';
benchmark_returns = T_log{:,1};
benchmark_dates = T_log.Properties.RowNames;
T_log_ottim = T_log(:, 2:end);

%% Execute all strategies simultaneously
[all_returns, all_weights, risk_metrics] = run_multi_risk_figarch_portfolio(...
    T_log_ottim, window_size, fallback_window, max_iter, 'all', custom_options);
%%
aligned_dates = all_returns.Properties.RowNames;
benchmark_series = benchmark_returns(ismember(benchmark_dates, aligned_dates));
all_returns.(benchmark_name) = benchmark_series;

%% Save in Excel
filename = 'Results_Portfolio_FIGARCH.xlsx';
strategies = all_returns.Properties.VariableNames;
dates = all_returns.Properties.RowNames;
returns_matrix = all_returns{:,:};

for i = 1:numel(strategies)
    sheet_name = strategies{i};
    T = table(dates, returns_matrix(:,i), 'VariableNames', {'Data', 'Return'});
    writetable(T, filename, 'Sheet', sheet_name);
end

for i = 1:(numel(strategies)-1)  
    W = all_weights{i};
    W = addvars(W, string(W.Properties.RowNames), 'Before', 1, 'NewVariableNames', 'Data');
    writetable(W, filename, 'Sheet', ['Weights_' strategies{i}]);
end

strategies_only = strategies(1:end-1);  

metrics_table = table(...
    risk_metrics.AnnualReturn*100, ...
    risk_metrics.AnnualVol*100, ...
    risk_metrics.Sharpe, ...
    risk_metrics.MaxDrawdown*100, ...
    risk_metrics.CVaR99*100, ...
    risk_metrics.Return_CVaR_Ratio, ...
    'VariableNames', {'AnnualReturn(%)', 'Volatility(%)', 'Sharpe', 'MaxDrawdown(%)', 'CVaR99(%)', 'Return/CVaR'});
metrics_table.Strategy = strategies_only';
metrics_table = movevars(metrics_table, 'Strategy', 'Before', 1);
writetable(metrics_table, filename, 'Sheet', 'Performance');


%% Cumulative returns graph
figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates), cumprod(1 + returns_matrix), 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('Cumulative returns - FIGARCH strategies');
ylabel('Portfolio Values');
xlabel('Data');
grid on;
saveas(gcf, 'figarch_cumulative_returns.png');
%% Drawdown graph for all strategies
drawdowns = zeros(size(returns_matrix));

for i = 1:numel(strategies)
    cum_returns = cumprod(1 + returns_matrix(:, i));
    peak = cummax(cum_returns);
    drawdowns(:, i) = (cum_returns ./ peak) - 1;
end

figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates), drawdowns, 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('Drawdown - FIGARCH strategies');
xlabel('Data');
ylabel('Drawdown (%)');
grid on;
saveas(gcf, 'figarch_drawdowns.png');

%% Rolling Sharpe Ratio graph
window_perf = 126;
rolling_sharpe = zeros(height(all_returns)-window_perf+1, numel(strategies));
for i = 1:numel(strategies)
    for t = window_perf:height(all_returns)
        data_window = all_returns{t-window_perf+1:t, i};
        rolling_sharpe(t-window_perf+1, i) = sqrt(252) * mean(data_window) / (std(data_window) + eps);
    end
end
figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates(window_perf:end)), rolling_sharpe, 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('Sharpe Ratio Rolling (126 days)');
xlabel('Data');
ylabel('Sharpe Ratio');
grid on;
saveas(gcf, 'figarch_rolling_sharpe.png');

%%
fprintf("\n=== DETAILED PERFORMANCE ===\n");

strategies = all_returns.Properties.VariableNames;

for i = 1:numel(strategies)
    strat_name = strategies{i};
    r_strategy = all_returns{:, i}; 
    metrics = compute_portfolio_metrics(r_strategy, 0);
    fprintf("\nPerformance - %s:\n", strrep(strat_name, '_', ' '));
    disp(metrics);
end
%%
for i = 1:numel(all_weights)
    figure('Position', [100, 100, 1200, 600]);
    area(datetime(all_weights{i}.Properties.RowNames), all_weights{i}{:,:});
    title(['Weight evolution - ' strrep(strategies{i}, '_', ' ')]);
    xlabel('Dates');
    ylabel('Weights (%)');
    legend(all_weights{i}.Properties.VariableNames, 'Location', 'eastoutside');
    grid on;
    saveas(gcf, ['figarch_weights_' strategies{i} '.png']);
end
%%
window_var = 126;
rolling_var = NaN(height(all_returns)-window_var+1, numel(strategies));
rolling_cvar = NaN(size(rolling_var));
alpha = custom_options.alpha;

for i = 1:numel(strategies)
    for t = window_var:height(all_returns)
        sample = all_returns{t-window_var+1:t, i};
        sorted = sort(sample);
        idx = max(1, ceil((1 - alpha) * length(sorted)));
        rolling_var(t-window_var+1, i) = -sorted(idx);
        rolling_cvar(t-window_var+1, i) = -mean(sorted(1:idx));
    end
end

figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates(window_var:end)), rolling_var, 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('VaR 99% Rolling - FIGARCH');
ylabel('VaR (%)');
xlabel('Data');
grid on;
saveas(gcf, 'figarch_rolling_var.png');

figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates(window_var:end)), rolling_cvar, 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('CVaR 99% Rolling - FIGARCH');
ylabel('CVaR (%)');
xlabel('Data');
grid on;
saveas(gcf, 'figarch_rolling_cvar.png');
%%
rolling_rcvar = NaN(size(rolling_var));
for i = 1:numel(strategies)
    for t = window_var:height(all_returns)
        sample = all_returns{t-window_var+1:t, i};
        mean_return = mean(sample);
        cvar_t = rolling_cvar(t-window_var+1, i);
        rolling_rcvar(t-window_var+1, i) = mean_return / abs(cvar_t + eps);
    end
end

figure('Position', [100, 100, 1200, 600]);
plot(datetime(dates(window_var:end)), rolling_rcvar, 'LineWidth', 1.5);
legend(strrep(strategies, '_', ' '), 'Location', 'best');
title('Return / CVaR Rolling - FIGARCH');
xlabel('Data');
ylabel('Return/CVaR');
grid on;
saveas(gcf, 'figarch_rolling_rcvar.png');

