# mk_motormap

The controlled step motor maps require output from 4 runs:
* reverse.lst
* forward.lst
* reverse-stage1.lst
* forward-stage1.lst

The output directories of these runs should be in the ./data directory. 
If you specify the directories on the commandline (see below), there can be
arbitrarily many directories in ./data/. If you want the program to 
automatically find the output directories, put only the four output
directories in ./data .

Then, the motor maps can be generated from the commandline as follows:
> ./mk_motormap
To set the paths to the output directories manually, add
> ./mk_motormap fwd_stage1=<dirname> rev_stage1=<dirname> fwd_stage2=<dirname> rev_stage2=<dirname>
where <dirname> is the full path to the output directory.

By default, mk_motormap will use ./xml_files/usedXMLFile.xml as the base
XML to write into. To change this, use
> ./mk_motormap xml=<path>

The motor map output (individual motor maps and associated figures) will 
be output to $savedir. The output motor map will also be called $savedir.xml.
By default, savedir will be ./motormaps/mmap_YYYY-MM-DD.
To change this:
> ./mk_motormap savedir=<dirname>




 