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
SimpleReturns = exp(LogReturns) - 1;
T_log = array2table(LogReturns, 'VariableNames', DataOnly.Properties.VariableNames);
T_log = addvars(T_log, T.Name, 'Before', 1, 'NewVariableNames', 'Name');
%% Cumulative return
CumReturns = cumprod(1 + LogReturns, 'omitnan');

figure;
plot(datenum(T_log.Name), CumReturns);
datetick('x', 'yyyy');
title('Cumulative returns of HFRX strategies');
xlabel('Years');
ylabel('Cumulative values');
legend(T_log.Properties.VariableNames(2:end), 'Location', 'best');
grid on;

%% Empirical analysis - ADF
nVars = size(CumReturns, 2);
adf_pvals = NaN(nVars,1);
for i = 1:nVars
    r = T_log{:, i+1};
    r = r(~isnan(r));
    if length(r) > 40
        [~, pVal] = adftest(r);
        adf_pvals(i) = pVal;
    end
end

ADF_Test_Results = table(adf_pvals, 'VariableNames', {'pValue'}, ...
    'RowNames', T_log.Properties.VariableNames(2:end));
disp('ADF test:');
disp(ADF_Test_Results);
%% Static and rolling Sharpe Ratio
rf = 0;
mu = mean(SimpleReturns, 'omitnan') * 252;
sigma = std(SimpleReturns, 'omitnan') * sqrt(252);
sharpe_ratio = (mu - rf) ./ sigma;
SharpeTable = table(T_log.Properties.VariableNames(2:end)', mu', sigma', sharpe_ratio', ...
    'VariableNames', {'Strategy', 'AnnualReturn', 'AnnualVolatility', 'SharpeRatio'});
SharpeTable = sortrows(SharpeTable, 'SharpeRatio', 'descend');

disp('Ranking Strategies by Sharpe Ratio:');
disp(SharpeTable);

%% Sharpe Ratio Rolling (1y)
window = 252;  
rollingSR = NaN(size(SimpleReturns));

for i = 1:nVars
    rolling_mean = movmean(SimpleReturns(:, i), window, 'omitnan');
    rolling_std = movstd(SimpleReturns(:, i), window, 'omitnan');
    rollingSR(:, i) = (rolling_mean - rf/252) ./ rolling_std;
end

figure;
plot(datenum(T_log.Name), rollingSR, 'LineWidth', 1.2);
datetick('x', 'yyyy');
title('Annual Sharpe Ratio (annual window)');
xlabel('Year'); ylabel('Sharpe Ratio');
legend(T_log.Properties.VariableNames(2:end), 'Location', 'best');
grid on;
yline(0, '--k', 'Sharpe = 0');  

%% Sharpe Ratio
figure;
b = bar(sharpe_ratio, 'FaceColor', 'flat');
colormap([1 0.3 0.3; 0.3 0.7 1]);  
colors = repmat([1 0.3 0.3], nVars, 1); 
colors(sharpe_ratio > 0, :) = repmat([0.3 0.7 1], sum(sharpe_ratio > 0), 1);  
b.CData = colors;

set(gca, 'XTick', 1:nVars);
set(gca, 'XTickLabel', T_log.Properties.VariableNames(2:end), 'XTickLabelRotation', 45);
title('Annual sharpe ratio');
ylabel('Sharpe Ratio');
set(gca, 'FontSize', 10);
grid on;

%% Sortino Ratio 


target_return = rf / 252; 
downside_dev = NaN(1, nVars);

for i = 1:nVars
    r = SimpleReturns(:, i);
    downside_diff = min(0, r - target_return);
    downside_dev(i) = sqrt(mean(downside_diff.^2, 'omitnan')) * sqrt(252);  
end

sortino_ratio = (mu - rf) ./ downside_dev;
SortinoTable = table(T_log.Properties.VariableNames(2:end)', mu', downside_dev', sortino_ratio', ...
    'VariableNames', {'Strategy', 'AnnualReturn', 'DownsideVolatility', 'SortinoRatio'});
SortinoTable = sortrows(SortinoTable, 'SortinoRatio', 'descend');

disp('Ranking Strategies by Sortino Ratio:');
disp(SortinoTable);

figure;
s = bar(sortino_ratio, 'FaceColor', 'flat');
colormap([1 0.3 0.3; 0.3 0.7 1]);

colors_sortino = repmat([1 0.3 0.3], nVars, 1);  % Default rosso
colors_sortino(sortino_ratio > 0, :) = repmat([0.3 0.7 1], sum(sortino_ratio > 0), 1);
s.CData = colors_sortino;

set(gca, 'XTick', 1:nVars);
set(gca, 'XTickLabel', T_log.Properties.VariableNames(2:end), 'XTickLabelRotation', 45);
title('Annual Sortino Ratio');
ylabel('Sortino Ratio');
set(gca, 'FontSize', 10);
grid on;
%% Max Drawdown 
nVars = size(CumReturns, 2);
maxDD = zeros(1, nVars);

for i = 1:nVars
    cum_curve = CumReturns(:, i);
    running_max = cummax(cum_curve);
    drawdown = (cum_curve - running_max) ./ running_max;
    maxDD(i) = min(drawdown);
end

DrawdownTable = table(T_log.Properties.VariableNames(2:end)', maxDD', ...
    'VariableNames', {'Strategy', 'MaxDrawdown'});

disp('Max Drawdown:');
disp(DrawdownTable);
%%
CumReturns = cumprod(1 + SimpleReturns, 'omitnan');
nVars = size(CumReturns, 2);
DrawdownSeries = NaN(size(CumReturns));

for i = 1:nVars
    cum_curve = CumReturns(:, i);
    if sum(~isnan(cum_curve)) < 200
        continue;
    end
    running_max = cummax(cum_curve);
    drawdown = (cum_curve - running_max) ./ running_max;
    DrawdownSeries(:, i) = drawdown;
end

% Plotting
figure;
plot(datetime(T_log.Name), DrawdownSeries, 'LineWidth', 1.2);
title('Rolling Drawdown');
xlabel('Data');
ylabel('Drawdown (%)');
legend(T_log.Properties.VariableNames(2:end), 'Location', 'best');
grid on;
ylim([-1 0]);

%% Risk-Return Map
mu = mean(SimpleReturns, 'omitnan') * 252;
sigma = std(SimpleReturns, 'omitnan') * sqrt(252);
sharpe = mu ./ sigma;

etichette_brevi = { ...
    'RelValArb', 'RelValFIConvArb', 'EqHedge', 'EventDriven', ...
    'MergerArb', 'GlobalHF', 'MacroCTA', 'MacroSys', ...
    'EqNeutral' };

figure;
scatter(sigma, mu, 1000 , sharpe, 'filled');
colorbar;
colormap jet;
xlabel('Annual volatility');
ylabel('Average annual return');
title('Risk return map by Sharpe Ratio');
grid on;
text(sigma, mu - 0.001, etichette_brevi, ...
     'FontSize', 14, ...
     'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'top');



%% Heatmap: risk-return
mu = mean(LogReturns, 'omitnan');
sigma = std(LogReturns, 'omitnan');

figure;
heatmap(T_log.Properties.VariableNames(2:end), {'Mean Return'}, mu,'Colormap',hot);
title('Average return per Strategy');

figure;
heatmap(T_log.Properties.VariableNames(2:end), {'Standard Deviation'}, sigma,'Colormap',hot);
title('Standard deviation by strategy');


%% Descriptive statistics 
Data = T_log{:, 2:end};  

SummaryStats = table;
SummaryStats.Mean = mean(Data, 'omitnan')';
SummaryStats.Std = std(Data, 'omitnan')';
SummaryStats.Min = min(Data, [], 'omitnan')';
SummaryStats.Max = max(Data, [], 'omitnan')';

% Skewness, kurtosis e Jarque-Bera
nVars = size(Data, 2);
skew = NaN(nVars, 1);
kurt = NaN(nVars, 1);
jbStat = NaN(nVars, 1);
jbPval = NaN(nVars, 1);

for i = 1:nVars
    validData = Data(:, i);
    validData = validData(~isnan(validData));
    
    s = skewness(validData);
    k = kurtosis(validData);
    n = length(validData);
    skew(i) = s;
    kurt(i) = k;

    
    jb = n/6 * (s^2 + (1/4)*(k - 3)^2);
    jbStat(i) = jb;
    jbPval(i) = 1 - chi2cdf(jb, 2);  % p-value
end

SummaryStats.Skewness = skew;
SummaryStats.Kurtosis = kurt;
SummaryStats.JBstat = jbStat;
SummaryStats.JBpval = jbPval;

SummaryStats.Properties.RowNames = T_log.Properties.VariableNames(2:end);

disp('Descriptive statstics:');
disp(SummaryStats);




%% Line Plot
figure;
plot(datenum(T_log.Name), T_log{:, 2:end});
datetick('x', 'yyyy');
legend(T_log.Properties.VariableNames(2:end), 'Location', 'best');
title('Log-returns');
xlabel('Year');
ylabel('Log return');
grid on;

%% Boxplot
figure;
boxplot(T_log{:, 2:end}, 'Labels', T_log.Properties.VariableNames(2:end));
title('Boxplot of LogReturns');
ylabel('log return');

%% Histograms
figure;
for i = 1:size(T_log, 2) - 1
    subplot(3, 3, i); 
    histogram(T_log{:, i+1}, 30);
    title(T_log.Properties.VariableNames{i+1});
    xlabel('log return');
    ylabel('Frequency');
end

%% Heatmap of correlations
CorrMatrix = corr(T_log{:, 2:end}, 'Rows', 'pairwise');
figure;
heatmap(T_log.Properties.VariableNames(2:end), ...
        T_log.Properties.VariableNames(2:end), ...
        CorrMatrix, ...
        'Colormap', hot, 'ColorbarVisible', 'on');
title('Correlation LogReturn');
%% rolling interactive dynamic correlation matrix 
window = 252;  
step = 50;   
dates = datetime(T_log.Name);
logrets = T_log{:, 2:end};
nAssets = size(logrets, 2);
assetNames = T_log.Properties.VariableNames(2:end);

start_idx = 1;
end_idx = start_idx + window - 1;

f = figure('Color', 'w');

while end_idx <= size(logrets, 1)
    
    sub_data = logrets(start_idx:end_idx, :);
    

    if any(sum(isnan(sub_data)) > window * 0.3)
        start_idx = start_idx + step;
        end_idx = start_idx + window - 1;
        continue;
    end


    R = corr(sub_data, 'Rows', 'pairwise');
    clf;
    heatmap(assetNames, assetNames, R, ...
        'Colormap', turbo, ...
        'ColorLimits', [-1 1], ...
        'ColorbarVisible', 'on');
    
    h = heatmap(assetNames, assetNames, R, ...
    'Colormap', turbo, ...
    'ColorLimits', [-1 1], ...
    'ColorbarVisible', 'on');

h.Title = sprintf('Rolling Correlation - %s to %s', ...
        datestr(dates(start_idx)), datestr(dates(end_idx)));

    
    drawnow;
    pause(0.3); 

    
    start_idx = start_idx + step;
    end_idx = start_idx + window - 1;
end


%% Scatter Matrix
figure;
plotmatrix(T_log{:, 2:end});
sgtitle('Scatter Matrix LogReturns');

%% Rolling volatility (30d)
window = 30;
RollingVol = movstd(T_log{:, 2:end}, window, 'omitnan');

figure;
plot(datenum(T_log.Name), RollingVol);
datetick('x', 'yyyy');
legend(T_log.Properties.VariableNames(2:end), 'Location', 'best');
title('Rolling Window (30d)');
xlabel('Years');
ylabel('Volatility');
grid on;
%% Engle's test
arch_pvals = NaN(nVars, 1);  
arch_stats = NaN(nVars, 1);  

for i = 1:nVars
    r = T_log{:, i+1};
    r = r(~isnan(r));  
    if length(r) > 40  
        [h, p, stat] = archtest(r, 'Lags', 5); 
        arch_pvals(i) = p;
        arch_stats(i) = stat;
    end
end


ARCH_Test_Results = table(arch_stats, arch_pvals, ...
    'VariableNames', {'TestStatistic', 'pValue'}, ...
    'RowNames', T_log.Properties.VariableNames(2:end));

disp('Engle test: ');
disp(ARCH_Test_Results);
%% =============================================== FIGARCH MODEL QMLE  ===================================================

J = 100;  
nVars = size(T_log, 2) - 1;


LogL_vals      = NaN(nVars, 1);
Theta_vals     = NaN(nVars, 5);   
ExitFlags      = NaN(nVars, 1);
Errors         = strings(nVars, 1);
sigma2_all     = cell(nVars, 1);
std_errors_all = cell(nVars, 1);
VCov_QMLE      = cell(nVars, 1);  
pvals_all      = cell(nVars, 1);  
JB_pvalues = NaN(nVars, 1);  


for i = 1:nVars
    try
        
        r = T_log{:, i+1};
        r = r(~isnan(r));

        
        try
            spec = garch(1,1);
            [~, ~, ~, param] = estimate(spec, r, 'Display', 'off');
            mu_init     = mean(r);
            omega_init  = param.Constant;
            alpha_init  = param.ARCH{1};
            beta_init   = param.GARCH{1};
            d_init      = 0.3;
        catch
            % Fallback
            mu_init     = mean(r);
            omega_init  = var(r) * 0.05;
            alpha_init  = 0.15;
            beta_init   = 0.7;
            d_init      = 0.2789;
        end

        
        phi_init = beta_init + (1 - beta_init) * alpha_init;
        theta0   = [mu_init; omega_init; phi_init; beta_init; d_init];

        
        [theta_hat, sigma2_hat, loglik_val, info] = stima_figarch_qmle(r, J, theta0);

        
        Theta_vals(i, :)     = theta_hat(:)';
        std_errors_all{i}    = info.std_errors;
        pvals_all{i}         = info.p_values;
        sigma2_all{i}        = sigma2_hat;
        resid_std = (r - theta_hat(1)) ./ sqrt(sigma2_hat);

        
        [~, p_jb] = jbtest(resid_std);
        JB_pvalues(i) = p_jb;

        VCov_QMLE{i}         = info.V_qmle;
        LogL_vals(i)         = loglik_val;
        ExitFlags(i)         = info.exitflag;

    catch ME
        Errors(i) = string(ME.message);
        disp("Error for strategy: " + T_log.Properties.VariableNames{i+1});
        disp(ME.message);
    end
end


SE_vals = NaN(nVars, 5);
for i = 1:nVars
    if ~isempty(std_errors_all{i})
        SE_vals(i, :) = std_errors_all{i}(:)';
    end
end


P_vals = NaN(nVars, 5);
for i = 1:nVars
    if ~isempty(pvals_all{i})
        P_vals(i, :) = pvals_all{i}(:)';
    end
end


Results_FIGARCH_QMLE = array2table([Theta_vals(:,1), SE_vals(:,1), P_vals(:,1), ...
                                     Theta_vals(:,2), SE_vals(:,2), P_vals(:,2), ...
                                     Theta_vals(:,3), SE_vals(:,3), P_vals(:,3), ...
                                     Theta_vals(:,4), SE_vals(:,4), P_vals(:,4), ...
                                     Theta_vals(:,5), SE_vals(:,5), P_vals(:,5), ...
                                     LogL_vals, ExitFlags], ...
    'VariableNames', {'mu','mu_SE','mu_pval','omega','omega_SE','omega_pval', ...
                      'phi','phi_SE','phi_pval','beta','beta_SE','beta_pval', ...
                      'd','d_SE','d_pval','LogLik','ExitFlag'}, ...
    'RowNames', T_log.Properties.VariableNames(2:end));


disp('QMLE FIGARCH(1,d,1) Parameters:');
disp(Results_FIGARCH_QMLE);

Results_JarqueBera = table(JB_pvalues, 'VariableNames', {'JB_pval'}, ...
    'RowNames', T_log.Properties.VariableNames(2:end));

disp('Jarque-Bera test on standardized reisduals:');
disp(Results_JarqueBera);



for i = 1:nVars
    fprintf('\nVariance-Covariance Matrix (QMLE) for %s:\n', T_log.Properties.VariableNames{i+1});
    disp(array2table(VCov_QMLE{i}, ...
        'VariableNames', {'mu','omega','phi1','beta1','d'}, ...
        'RowNames',      {'mu','omega','phi1','beta1','d'}));
end
%% PLOTTING 
window = 30;
nVars = size(T_log, 2) - 1;
sigma2_all = sigma2_all(:);

f = figure('Color', 'w', 'Position', [100 100 1400 1000]); 

for i = 1:nVars
    try
        r = T_log{:, i+1};
        dates = T_log.Name;

        
        r = r(~isnan(r));
        dates = dates(~isnan(T_log{:, i+1}));

        sigma2 = sigma2_all{i};
        if isempty(sigma2) || all(isnan(sigma2))
            warning("sigma2 not available %s", T_log.Properties.VariableNames{i+1});
            continue;
        end

       
        if length(sigma2) < length(r)
            sigma2 = [NaN(length(r)-length(sigma2), 1); sigma2];
        elseif length(sigma2) > length(r)
            sigma2 = sigma2(end - length(r) + 1 : end);
        end

        
        vol_figarch = sqrt(sigma2) * sqrt(252) * 100;
        vol_real = movstd(r, [window-1 0]) * sqrt(252) * 100;
        vol_figarch_smooth = movmean(vol_figarch, 3);

      
        subplot(3, 3, i);
        plot(dates, vol_figarch_smooth, 'b-', 'LineWidth', 1.1); hold on;
        plot(dates, vol_real, 'r-', 'LineWidth', 1.1);
        title(strrep(T_log.Properties.VariableNames{i+1}, '_', '\_'), 'Interpreter', 'tex');
        grid on;
        xlim([dates(1) dates(end)]);
        ylim([0 max(max(vol_figarch_smooth, [], 'omitnan'), max(vol_real, [], 'omitnan')) * 1.1]);
        if i > 6
            xlabel('Date');
        end
        if mod(i,3) == 1
            ylabel('Volatility (%)');
        end
        if i == 1
            legend({'FIGARCH', sprintf('Realized (%d days)', window)}, 'Location', 'northeast');
        end

    catch ME
        warning("Error in the strategy %s: %s", T_log.Properties.VariableNames{i+1}, ME.message);
    end
end

sgtitle('Estimated vs Realized Volatility – All Strategies', 'FontSize', 14, 'FontWeight', 'bold');


%% VAR in-sample 
alpha = 0.01;
nu = 8; %approx.
T1 = 1;
T10 = 10;

strategies = T_log.Properties.VariableNames(2:end);
var_results = [];

for i = 1:nVars
    try
        mu = Theta_vals(i,1);
        sigma2 = sigma2_all{i};
        if isempty(sigma2)
            continue;
        end
        sigma_t = sqrt(sigma2);
        sigma_last = sigma_t(end);

        r = T_log{:, i+1};
        r = r(~isnan(r));
        residui = r - mu;
        S = skewness(residui, 0);
        K = kurtosis(residui, 0) - 3; 

        
        z_alpha = norminv(alpha);
        t_alpha = tinv(alpha, nu);
        z = z_alpha;
        z_cf = z ...
             + (1/6)*((z^2) - 1)*S ...
             + (1/24)*((z^3) - 3*z)*K ...
             - (1/36)*((2*z^3) - 5*z)*S^2;

        
        std1 = sqrt(T1) * sigma_last;
        VaR1_gaussian = T1 * mu + z_alpha * std1;
        VaR1_student  = T1 * mu + t_alpha * std1;
        VaR1_cf       = T1 * mu + z_cf    * std1;

        ES1_gaussian = T1 * mu - std1 * normpdf(z_alpha) / alpha;
        ES1_student  = T1 * mu - std1 * tpdf(t_alpha, nu) * (nu + t_alpha^2) / ((nu - 1) * alpha);
        ES1_cf       = T1 * mu - std1 * normpdf(z_cf) / alpha;

        
        std10 = sqrt(T10) * sigma_last;
        VaR10_gaussian = T10 * mu + z_alpha * std10;
        VaR10_student  = T10 * mu + t_alpha * std10;
        VaR10_cf       = T10 * mu + z_cf    * std10;

        ES10_gaussian = T10 * mu - std10 * normpdf(z_alpha) / alpha;
        ES10_student  = T10 * mu - std10 * tpdf(t_alpha, nu) * (nu + t_alpha^2) / ((nu - 1) * alpha);
        ES10_cf       = T10 * mu - std10 * normpdf(z_cf) / alpha;

        
        var_results = [var_results; {
            strategies{i}, ...
            VaR1_gaussian, VaR1_student, VaR1_cf, ...
            ES1_gaussian, ES1_student, ES1_cf, ...
            VaR10_gaussian, VaR10_student, VaR10_cf, ...
            ES10_gaussian, ES10_student, ES10_cf, ...
            sigma_last, S, K
        }];

    catch ME
        fprintf("Error during the computation of VaR/ES per %s: %s\n", strategies{i}, ME.message);
        continue
    end
end

figarch_var_full_table = cell2table(var_results, ...
    'VariableNames', {'Strategia', ...
    'VaR1_Gaussian', 'VaR1_Student', 'VaR1_CF', ...
    'ES1_Gaussian', 'ES1_Student', 'ES1_CF', ...
    'VaR10_Gaussian', 'VaR10_Student', 'VaR10_CF', ...
    'ES10_Gaussian', 'ES10_Student', 'ES10_CF', ...
    'Volatility', 'Skewness', 'Kurtosis'});


numVars = figarch_var_full_table(:, 2:end) .* 100;  
numVars = varfun(@(x) round(x, 7), numVars);
figarch_var_full_table = [figarch_var_full_table(:,1), numVars];

disp(" Complete Results for VaR and ES (%) :");
disp(figarch_var_full_table);
alpha = 0.01;
nu = 8;

strategies = T_log.Properties.VariableNames(2:end);
z_alpha = norminv(alpha);
t_alpha = tinv(alpha, nu);

for i = 1:nVars
    try
        mu = Theta_vals(i, 1);
        sigma2 = sigma2_all{i};
        if isempty(sigma2)
            continue;
        end

        sigma_t = sqrt(sigma2); 
        r = T_log{:, i+1};      
        r = r(~isnan(r));
        t_axis = T_log.Name(~isnan(T_log{:, i+1}));  

        residui = r - mu;
        S = min(max(skewness(residui, 0), -3), 3);
        K = min(max(kurtosis(residui, 0) - 3, 0), 30);
        z_cf = z_alpha ...
             + (1/6)*((z_alpha^2) - 1)*S ...
             + (1/24)*((z_alpha^3) - 3*z_alpha)*K ...
             - (1/36)*((2*z_alpha^3) - 5*z_alpha)*S^2;

        
        VaR_gaussian = mu + z_alpha * sigma_t;
        VaR_student  = mu + t_alpha * sigma_t;
        VaR_cf       = mu + z_cf     * sigma_t;
        figure('Name', strategies{i}, 'Color', 'w', 'Position', [100, 100, 1000, 400]);
        hold on
        plot(t_axis, r, 'k', 'LineWidth', 1.2, 'DisplayName', 'Rendimento reale');
        plot(t_axis, VaR_gaussian, '--b', 'LineWidth', 1.2, 'DisplayName', 'VaR Gaussian (1%)');
        plot(t_axis, VaR_student, '--g', 'LineWidth', 1.2, 'DisplayName', 'VaR Student (1%)');
        plot(t_axis, VaR_cf, '--r', 'LineWidth', 1.2, 'DisplayName', 'VaR Cornish-Fisher (1%)');

        
        area(t_axis, VaR_gaussian, 'FaceColor', 'blue', 'FaceAlpha', 0.05, 'EdgeAlpha', 0);

        title(['VaR  - ', strategies{i}], 'FontSize', 14);
        xlabel('Date');
        ylabel('Return');
        legend('show', 'Location', 'best');
        grid on
        xlim([t_axis(1) t_axis(end)]);
        hold off

    catch ME
        fprintf("Error in the graph for %s: %s\n", strategies{i}, ME.message);
        continue
    end
end



%%
figure;
histogram(resid_std, 30); title('Histogram of standardized residuals');
figure;
qqplot(resid_std); title('Q-Q Plot');
figure;
autocorr(resid_std); title('ACF of residuals');
figure;
autocorr(resid_std); title('ACF of squared residuals');
%%
% LR Test for d=0 (FIGARCH vs GARCH)
nVars = size(T_log, 2) - 1;
LR_stats = NaN(nVars, 1);
LR_pvals = NaN(nVars, 1);
LogL_GARCH = NaN(nVars, 1);

for i = 1:nVars
    try
        r = T_log{:, i+1};
        r = r(~isnan(r));
        
        
        if ~isnan(LogL_vals(i))
        
            mu_init = Theta_vals(i,1);
            omega_init = Theta_vals(i,2);
            phi_init = Theta_vals(i,3);
            beta_init = Theta_vals(i,4);
            
        
            theta0_garch = [mu_init; omega_init; phi_init; beta_init; 0];
            
          
            % Create a temporary copy of stima_figarch_qmle with a modification
            % to force d=0 
            create_garch_estimator(J); % This creates stima_garch_qmle
            
            % Run it
            [~, ~, loglik_garch, ~] = stima_garch_qmle(r, J, theta0_garch);
            LogL_GARCH(i) = loglik_garch;
            
            % Calculate LR statistic
            LR_stats(i) = 2 * (LogL_vals(i) - LogL_GARCH(i));
            
            
            LR_pvals(i) = 1 - chi2cdf(LR_stats(i), 1);
            
            
            disp(['Completed LR test for: ', T_log.Properties.VariableNames{i+1}, ...
                ' - LR stat: ', num2str(LR_stats(i)), ', p-value: ', num2str(LR_pvals(i))]);
        end
        
    catch ME
        disp("Error on LR test for: " + T_log.Properties.VariableNames{i+1});
        disp(ME.message);
    end
end


LR_Results = array2table([LR_stats, LR_pvals, LogL_vals, LogL_GARCH], ...
    'VariableNames', {'LR_Statistic', 'p_Value', 'LogL_FIGARCH', 'LogL_GARCH'}, ...
    'RowNames', T_log.Properties.VariableNames(2:end));


disp('Likelihood Ratio Test Results for H0: d=0 (FIGARCH vs GARCH):');
disp(LR_Results);


valid_idx = ~isnan(LR_pvals);
if sum(valid_idx) > 0
    alpha = 0.01; 
    significant = LR_pvals(valid_idx) < alpha;
    disp(['Reject H0 (d=0) at 5% significance level for ', num2str(sum(significant)), ...
          ' out of ', num2str(sum(valid_idx)), ' valid series.']);
    
    
    if any(significant)
        sig_indices = find(valid_idx);
        sig_indices = sig_indices(significant);
        disp('Series with significant long memory (d≠0):');
        disp(T_log.Properties.VariableNames(1+sig_indices)');
    end
else
    disp('No valid test results obtained.');
end


function create_garch_estimator(J_val)

    fid = fopen('stima_garch_qmle.m', 'w');

    fprintf(fid, 'function [theta_hat, sigma2_hat, loglik_val, info] = stima_garch_qmle(r, J, theta0)\n');
    fprintf(fid, '    %% Modified version of stima_figarch_qmle with d fixed at 0\n\n');

    fprintf(fid, '    J = %d; %% Use the value provided\n\n', J_val);

    fprintf(fid, '    %% Ensure d=0 (GARCH restriction)\n');
    fprintf(fid, '    theta0(5) = 0;\n\n');

    fprintf(fid, '    %% Bounds with d fixed at 0\n');
    fprintf(fid, '    LB = [-Inf, 1e-12, 0, 0, 0];\n');
    fprintf(fid, '    UB = [Inf, Inf, 0.999, 0.999, 0];\n\n');

    fprintf(fid, '    %% Constraints for stability\n');
    fprintf(fid, '    nonlcon = @(theta) garch_constraints(theta);\n\n');

    fprintf(fid, '    %% Objective function\n');
    fprintf(fid, '    objfun = @(theta) loglik_garch(theta, r, J);\n\n');

    fprintf(fid, '    %% Optimization options\n');
    fprintf(fid, '    options = optimoptions(''fmincon'', ...\n');
    fprintf(fid, '        ''Display'', ''off'', ...\n');
    fprintf(fid, '        ''Algorithm'', ''interior-point'', ...\n');
    fprintf(fid, '        ''MaxIterations'', 1000, ...\n');
    fprintf(fid, '        ''OptimalityTolerance'', 1e-6);\n\n');

    fprintf(fid, '    %% Optimization with GARCH restriction\n');
    fprintf(fid, '    try\n');
    fprintf(fid, '        [theta_hat, loglik_val, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], LB, UB, nonlcon, options);\n');
    fprintf(fid, '    catch ME\n');
    fprintf(fid, '        fprintf(''Optimization error: %%s\\n'', ME.message);\n');
    fprintf(fid, '        theta_hat = NaN(5,1);\n');
    fprintf(fid, '        loglik_val = NaN;\n');
    fprintf(fid, '        exitflag = -99;\n');
    fprintf(fid, '        output = struct();\n');
    fprintf(fid, '    end\n\n');

    fprintf(fid, '    %% Ensure d=0 in result\n');
    fprintf(fid, '    theta_hat(5) = 0;\n\n');

    fprintf(fid, '    %% Calculate variance\n');
    fprintf(fid, '    sigma2_hat = garch_variance(r, theta_hat, J);\n\n');

    fprintf(fid, '    %% Create info structure\n');
    fprintf(fid, '    info = struct();\n');
    fprintf(fid, '    info.exitflag = exitflag;\n');
    fprintf(fid, '    info.output = output;\n');
    fprintf(fid, '    info.parameters = struct(''mu'', theta_hat(1), ''omega'', theta_hat(2), ...\n');
    fprintf(fid, '                           ''phi1'', theta_hat(3), ''beta1'', theta_hat(4), ''d'', 0);\n');
    fprintf(fid, 'end\n\n');

    fprintf(fid, 'function sigma2 = garch_variance(r, theta, J)\n');
    fprintf(fid, '    %% GARCH variance calculation (FIGARCH with d=0)\n');
    fprintf(fid, '    mu = theta(1);\n');
    fprintf(fid, '    omega = theta(2);\n');
    fprintf(fid, '    alpha = theta(3); %% phi in FIGARCH notation\n');
    fprintf(fid, '    beta = theta(4);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    T = length(r);\n');
    fprintf(fid, '    e = r - mu;\n');
    fprintf(fid, '    e2 = e.^2;\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Initialize\n');
    fprintf(fid, '    sigma2 = zeros(T,1);\n');
    fprintf(fid, '    sigma2(1) = var(r);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% Standard GARCH(1,1) recursion\n');
    fprintf(fid, '    for t = 2:T\n');
    fprintf(fid, '        sigma2(t) = omega + alpha * e2(t-1) + beta * sigma2(t-1);\n');
    fprintf(fid, '        if sigma2(t) <= 1e-6\n');
    fprintf(fid, '            sigma2(t) = 1e-6;\n');
    fprintf(fid, '        end\n');
    fprintf(fid, '    end\n');
    fprintf(fid, 'end\n\n');

    fprintf(fid, 'function negLL = loglik_garch(theta, r, J)\n');
    fprintf(fid, '    %% Log-likelihood function for GARCH model\n');
    fprintf(fid, '    try\n');
    fprintf(fid, '        sigma2 = garch_variance(r, theta, J);\n');
    fprintf(fid, '        mu = theta(1);\n');
    fprintf(fid, '        e = r - mu;\n');
    fprintf(fid, '        \n');
    fprintf(fid, '        %% Calculate log-likelihood\n');
    fprintf(fid, '        valid_idx = ~isnan(sigma2) & (sigma2 > 0);\n');
    fprintf(fid, '        loglik = -0.5 * log(2*pi) - 0.5 * log(sigma2(valid_idx)) - 0.5 * (e(valid_idx).^2) ./ sigma2(valid_idx);\n');
    fprintf(fid, '        \n');
    fprintf(fid, '        negLL = -sum(loglik);\n');
    fprintf(fid, '        \n');
    fprintf(fid, '        if isnan(negLL) || isinf(negLL)\n');
    fprintf(fid, '            negLL = 1e10;\n');
    fprintf(fid, '        end\n');
    fprintf(fid, '    catch\n');
    fprintf(fid, '        negLL = 1e10;\n');
    fprintf(fid, '    end\n');
    fprintf(fid, 'end\n\n');

    fprintf(fid, 'function [c, ceq] = garch_constraints(theta)\n');
    fprintf(fid, '    %% Constraints for GARCH stability\n');
    fprintf(fid, '    alpha = theta(3);\n');
    fprintf(fid, '    beta = theta(4);\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    %% GARCH stationarity constraint\n');
    fprintf(fid, '    c = alpha + beta - 0.999;\n');
    fprintf(fid, '    ceq = [];\n');
    fprintf(fid, 'end\n');
    
    fclose(fid);
end