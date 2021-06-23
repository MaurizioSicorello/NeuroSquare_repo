negAffect = fmri_data('Rating_Weights_LOSO_2.nii')

negAffect.dat = negAffect.dat *100;

caxis([-1 1])
t = surface(negAffect);



drawnow, snapnow

render_on_surface(negAffect, [], 'clim', [-1 1]);

my_display_obj = canlab_results_fmridisplay(negAffect, 'cmaprange', [-1 1]);

hist(negAffect.dat)

canlab_results_fmridisplay(negAffect, 'cmaprange', [-1 1])