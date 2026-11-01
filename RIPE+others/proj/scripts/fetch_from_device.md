# Moving the datasets into the compute workspace

The archives live on the user's D: drive. The device VM splits them into
<=380 MB chunks which are then staged into the cloud workspace and re-joined:

    # device VM
    cd ~/mnt/RIPE+others/data && mkdir -p _chunks
    split -b 380m hpatches-sequences-release.tar.gz _chunks/hp.part.
    split -b 380m megadepth1500.zip               _chunks/md.part.
    # workspace
    cat hp.part.* > hpatches-sequences-release.tar.gz && tar xzf ...
    cat md.part.* > megadepth1500.zip && unzip ...
