function make_charts()
% XRD embedded contest report figures.
% Generated with MATLAB R2024b; figures are intentionally data-dense and
% visually aligned with the CIMC reference report style.

close all;
S = xrd_style();
outdir = fileparts(mfilename('fullpath'));

make_metric_landscape(S, outdir);
make_latency_ladder(S, outdir);
make_completion_matrix(S, outdir);

end

function S = xrd_style()
S.navy   = [31 59 87]/255;
S.accent = [46 125 242]/255;
S.teal   = [15 169 164]/255;
S.amber  = [243 165 27]/255;
S.orange = [242 104 42]/255;
S.green  = [23 166 90]/255;
S.purple = [124 58 237]/255;
S.red    = [239 68 68]/255;
S.grey   = [117 131 149]/255;
S.lgrey  = [220 229 239]/255;
S.ink    = [38 48 60]/255;
S.font   = 'Microsoft YaHei';
end

function prep(fig, w, h)
set(fig, 'Color', 'w', 'Units', 'pixels', 'Position', [80 80 w h]);
end

function save2(fig, outdir, stem)
exportgraphics(fig, fullfile(outdir, [stem '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(outdir, [stem '.png']), 'Resolution', 220);
end

function make_metric_landscape(S, outdir)
fig = figure('Name', 'xrd metric landscape'); prep(fig, 1180, 620);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

labels = {'AI 服务线','CPU 本地模型','BPU LLM slot','表征轻量模型','OpenAPI paths','实测 PL 行','论文 chunks'};
values = [5, 4, 5, 5, 89, 67, 25228];
colors = [S.accent; S.purple; S.orange; S.teal; S.green; S.amber; S.navy];

ax = nexttile;
barh(ax, log10(values), 0.65, 'FaceColor', 'flat', 'CData', colors);
set(ax, 'YTick', 1:numel(labels), 'YTickLabel', labels, 'YDir', 'reverse', 'FontName', S.font, 'FontSize', 11, 'XColor', S.ink, 'YColor', S.ink);
xlabel(ax, 'log10(数量)', 'FontName', S.font, 'FontWeight', 'bold');
title(ax, '工程资产数量跨多个量级', 'FontName', S.font, 'FontSize', 15, 'FontWeight', 'bold', 'Color', S.navy);
grid(ax, 'on'); ax.GridLineStyle=':'; ax.GridColor=S.lgrey;
for i = 1:numel(values)
    text(log10(values(i)) + 0.04, i, format_value(values(i)), 'FontName', S.font, 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', S.ink, 'VerticalAlignment', 'middle');
end
box(ax, 'off');

ax = nexttile;
axis(ax, 'off'); hold(ax, 'on');
title(ax, '报告主线：预测、执行、表征、回流', 'FontName', S.font, 'FontSize', 15, 'FontWeight', 'bold', 'Color', S.navy);
items = {
    '材料预测', 'host / dopant / 浓度 -> verdict / 风险 / 烧结建议', S.accent;
    '具身感知', 'LiDAR / 深度 / SLAM / Lab-FSD shadow 在线', S.orange;
    '工位执行', 'arm01 视觉门控取袋 + F407 瓶子动作降级演示', S.teal;
    '表征分析', 'XRD 与 PL 的视觉线、数值线进入 AI 脑', S.green;
    '证据回流', 'actuals / failure library / SHA hash 链 / 公网站', S.purple;
    };
y = 0.88;
for i = 1:size(items,1)
    rectangle(ax, 'Position', [0.04 y-0.08 0.92 0.115], 'Curvature', 0.08, 'FaceColor', [1 1 1], 'EdgeColor', items{i,3}, 'LineWidth', 1.8);
    rectangle(ax, 'Position', [0.04 y-0.08 0.025 0.115], 'Curvature', 0.08, 'FaceColor', items{i,3}, 'EdgeColor', items{i,3});
    text(ax, 0.09, y-0.012, items{i,1}, 'FontName', S.font, 'FontSize', 12.5, 'FontWeight', 'bold', 'Color', items{i,3});
    text(ax, 0.29, y-0.012, items{i,2}, 'FontName', S.font, 'FontSize', 10.5, 'Color', S.ink);
    y = y - 0.16;
end
xlim(ax,[0 1]); ylim(ax,[0 1]);

sgtitle('XRD 嵌入式项目工程指标总览', 'FontName', S.font, 'FontSize', 18, 'FontWeight', 'bold', 'Color', S.navy);
save2(fig, outdir, 'fig_xrd_metrics');
end

function make_latency_ladder(S, outdir)
fig = figure('Name', 'xrd latency ladder'); prep(fig, 1160, 620);

labels = {'Fly-MB MBON est','Fly-MB KC est','Lab-FSD BPU obs','BPU MLP obs','BPU Qwen2 0.5B','大 slot 工程展示','完整云/本地评审'};
ms = [0.0118, 0.0533, 1.722, 2.0, 553, 75000, 30000];
types = {'hb mapper','hb mapper','board obs','board obs','board obs','swap-load','SSE verdict'};
colors = [S.purple; S.purple; S.orange; S.green; S.accent; S.grey; S.navy];

ax = axes(fig);
barh(ax, log10(ms), 0.62, 'FaceColor', 'flat', 'CData', colors);
set(ax, 'YTick', 1:numel(labels), 'YTickLabel', labels, 'YDir', 'reverse', 'FontName', S.font, 'FontSize', 11, 'XColor', S.ink, 'YColor', S.ink);
xlabel(ax, 'log10(毫秒)', 'FontName', S.font, 'FontWeight', 'bold');
title(ax, '不同能力位延迟跨度：微秒级 BPU 头到分钟级大 slot 展示', 'FontName', S.font, 'FontSize', 16, 'FontWeight', 'bold', 'Color', S.navy);
grid(ax, 'on'); ax.GridLineStyle=':'; ax.GridColor=S.lgrey; box(ax, 'off');
for i = 1:numel(ms)
    if ms(i) < 1
        val = sprintf('%.1f us', ms(i)*1000);
    elseif ms(i) < 1000
        val = sprintf('%.1f ms', ms(i));
    else
        val = sprintf('%.1f s', ms(i)/1000);
    end
    text(log10(ms(i)) + 0.08, i, sprintf('%s  |  %s', val, types{i}), 'FontName', S.font, 'FontSize', 10.2, 'FontWeight', 'bold', 'Color', S.ink, 'VerticalAlignment', 'middle');
end
note = '注：est 为 hb_mapper 编译估计；obs 为工程观测。BPU LLM slot 用作 verdict probe / 工程展示，不宣传为长文本实时生成。';
text(ax, min(xlim(ax)), numel(labels)+0.9, note, 'FontName', S.font, 'FontSize', 10.5, 'Color', S.grey);
save2(fig, outdir, 'fig_xrd_latency');
end

function make_completion_matrix(S, outdir)
fig = figure('Name', 'xrd completion matrix'); prep(fig, 1180, 640);

rows = {'AI 脑五服务线','材料预测 evidence','XRD/PL 表征分析','SLAM 占据栅格','Lab-FSD shadow','arm01 取袋闭环','F407 瓶子动作','公网证据平台','Nav2 全自主闭环','升降台完整限位','双臂十阶段全闭环'};
cols = {'已实测','初赛降级/Shadow','后续补齐'};
M = [
    1 0 0;
    1 0 0;
    1 0 0;
    1 0 0;
    1 1 0;
    1 0 0;
    1 1 1;
    1 0 0;
    0 1 1;
    0 1 1;
    0 1 1;
    ];

ax = axes(fig, 'Position', [0.25 0.18 0.70 0.68]);
hold(ax, 'on');
xlim(ax, [0.5 3.5]); ylim(ax, [0.5 numel(rows)+0.5]);
set(ax, 'YDir', 'reverse', 'XTick', 1:numel(cols), 'XTickLabel', cols, ...
    'YTick', 1:numel(rows), 'YTickLabel', rows, 'FontName', S.font, ...
    'FontSize', 11, 'XColor', S.ink, 'YColor', S.ink, 'TickLength', [0 0]);
title(ax, '完成状态矩阵：把“已实测”和“安全降级/后续补齐”分开写', ...
    'FontName', S.font, 'FontSize', 16, 'FontWeight', 'bold', 'Color', S.navy);
for r = 1:size(M,1)
    for c = 1:size(M,2)
        rectangle(ax, 'Position', [c-0.45 r-0.36 0.90 0.72], 'Curvature', 0.12, ...
            'FaceColor', [0.97 0.98 0.99], 'EdgeColor', S.lgrey, 'LineWidth', 0.8);
        if M(r,c) > 0
            if c == 1
                color = S.green; txt = '实测';
            elseif c == 2
                color = S.amber; txt = '降级';
            else
                color = S.purple; txt = '待补';
            end
            rectangle(ax, 'Position', [c-0.34 r-0.23 0.68 0.46], 'Curvature', 0.30, ...
                'FaceColor', color, 'EdgeColor', 'none');
            text(ax, c, r, txt, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontName', S.font, 'FontWeight', 'bold', 'FontSize', 10.5, 'Color', 'w');
        end
    end
end
for c = 1.5:1:2.5
    plot(ax, [c c], [0.5 numel(rows)+0.5], '-', 'Color', [0.90 0.93 0.96], 'LineWidth', 1.2);
end
box(ax, 'off');
annotation(fig, 'textbox', [0.25 0.055 0.70 0.06], 'String', ...
    '报告原则：有真机证据的写“已实测”；算法在线但未接管执行的写“shadow/降级”；硬件保护未补齐的写“后续补齐”。', ...
    'FontName', S.font, 'FontSize', 10.5, 'Color', S.grey, 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center');
save2(fig, outdir, 'fig_xrd_completion_matrix');
end

function s = format_value(v)
if v >= 1000
    s = regexprep(sprintf('%.0f', v), '\B(?=(\d{3})+(?!\d))', ',');
else
    s = sprintf('%.0f', v);
end
end
