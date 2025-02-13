# `iodaconv`
Wrapper to JCSDA tools to convert obs files to ioda, concatenate and superob.
It is in a usable format but not exactly user friendly (yet? do we care?). It will only
work on Orion and is pointing to my own build of `soca-science` and `ioda-converters`.
TODO: Make it user friendly

- **godas_obs2ioda.sbatch**
Hard wired for Orion, meant for the `sst` files. This script will send 1 job that will request 1 node and assume 40 cores will be
available within that node.

Usage and setup:

```console
mkdir scratch                                                   # Create a `scratch` somewhere
cd scratch
ln -s <path to godas repo>/obsproc/iodaconv/godas_obs2ioda.* .  # link the shell scripts ... ugly, I know
mkdir data                                                      # link the source of observations. For example, for sst.avhrr_l3u.nesdis
ln -s <path to source of obs>/sst.avhrr_l3u.nesdis data/sst.avhrr_l3u.nesdis
./godas_obs2ioda.sbatch -s 20190101 -e 20190102 -t 00:10:00     # start a conversion job
```

- **godas_obscat.sh**
TODO (Guillaume): Probably should be here, move somewhere else
In progress, will concatenate files just don't use it just yet
