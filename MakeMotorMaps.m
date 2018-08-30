% Matlab wrapper for python scripts to generate controlled step
% motor maps.


% === Python set-up

% pyversion ( '/Users/kadofong/env/bin/python' ) % If your preferred python
% installation is not the *system* python installation, redirect here.
%   Note: The Matlab path is not the same as $PATH, so make sure that this
%   is set if you usually use e.g. a virtualenvironment

py.importlib.import_module ( 'mk_motormap' )
py.modelcobras.mk_motormap.main ()
