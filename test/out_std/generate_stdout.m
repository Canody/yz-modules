% FFT 图像旋转
rotdeg = 32;
im_rot = fftImAffine.fftImRotate(im, rotdeg);
save('./out_std/rot_im.mat', "im_rot", "rotdeg")

% FFT Bin
bin_scale = 0.55;
im_bin = fftImAffine.fftImBin(im, bin_scale);
save('./out_std/bin_im.mat', 'im_bin', "bin_scale")

% FFT 图像移动
shift_pixel = [5.2,-10.73];
im_shift = fftImAffine.fftImShift(im, shift_pixel);
save('./out_std/shift_im.mat', 'im_shift', 'shift_pixel')

% FFT 图像放大
expand_scale = 2.24;
im_expand = fftImAffine.fftImExpand(im, expand_scale);
save('./out_std/expand_im.mat', 'im_expand', 'expand_scale')

% FFT 图像收缩
shrink_scale = 1.86;
im_shrink = fftImAffine.fftImShrink(im, shrink_scale);
save('./out_std/shrink_im.mat', 'im_shrink', 'shrink_scale')

% FFT 图像 shear
sheardim = 1;  % ! IN PYTHON, BE 0 ! %
shfractor = 0.43;
im_shear = fftImAffine.fftImShear(im, sheardim, shfractor);
save('./out_std/shear_im.mat', 'im_shear', 'sheardim', 'shfractor')

f = figure;
f.Position = [100,100,627,317];

tiledlayout(2,4, 'TileSpacing', 'none', 'Padding', 'tight')
nexttile;
hold on; imagesc(im); text(0,0,'Raw', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_rot); text(0,0,'Rot 32 deg', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_bin); text(0,0,'Bin 0.55', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_shift); text(0,0,'Shift 5.2 -10.73', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_expand); text(0,0,'Expand 2.24', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_shrink); text(0,0,'Shrink 1.86', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

nexttile;
hold on; imagesc(im_shear); text(0,0,'Shear dim1 0.43', 'VerticalAlignment', 'top', 'Color', 'white'); hold off
axis image off; colormap gray
ax = gca; ax.YDir = 'reverse';

exportgraphics(gcf, './out_std/out_std.png', 'Resolution', 600)

