% Matlab wrapper for python scripts to generate controlled step
% motor maps.


% === Python set-up

%pyversion ( '/Users/kadofong/env/bin/python' ) % If your preferred python
% installation is not the *system* python installation, redirect here.
%   Note: The Matlab path is not the same as $PATH, so make sure that this
%   is set if you usually use e.g. a virtualenvironment

specdir = './data/18_07_18_15_30_34_forward-stage2/Log'

py.importlib.import_module ( 'modelcobras' )
py.modelcobras.control.moveto_stage1testing ( specdir )