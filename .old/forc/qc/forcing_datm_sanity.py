#!/usr/bin/env python3
                           #hyun-chul.lee@noaa.gov
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import argparse
import yaml

def plot_zonal_ave(data1zave,data1zmax,data1zmin,lats1,data2zave,data2zmax,data2zmin,lats2,plotpath,v,sdate,d1nm,d2nm,ne):
    #-- 
    mindt1=np.amin(data1zmin)
    mindt2=np.amin(data2zmin)
    len1=lats1.size
    len2=lats2.size
    mindt=min([mindt1,mindt2])
    
    maxdt1=np.amax(data1zmax)
    maxdt2=np.amax(data2zmax)
    maxdt=max([maxdt1,maxdt2])

    x1lat=np.reshape(lats1,len1)
    y1ave=np.reshape(data1zave,len1)
    y1min=np.reshape(data1zmin,len1)
    y1max=np.reshape(data1zmax,len1)
    x2lat=np.reshape(lats2,len2)
    y2ave=np.reshape(data2zave,len2)
    y2min=np.reshape(data2zmin,len2)
    y2max=np.reshape(data2zmax,len2)

    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    pn = ne + 1
    plt.subplot(9,3,pn)
    plt.plot(lats1,y1ave,'b')
    plt.plot(lats2,y2ave,'r')
    plt.plot(lats1,y1max,'b--')
    plt.plot(lats2,y2max,'r--')
    plt.plot(lats1,y1min,'b:')
    plt.plot(lats2,y2min,'r:')
    plt.xlim(-90.0,90.0)
    plt.ylim(mindt,maxdt)
    plt.title(v,fontsize=7,y=0.8)
    plt.yticks(fontsize=6)
    plt.xticks(fontsize=6)

    if (pn == 1):
        plt.suptitle('Zonal DATM of '+d1nm+'(blue) and '+d2nm+'(red) at '+sdate+' in Ave(line), Max(dashed), Min(dotted)',fontsize=8)

    if (pn < 25):
        plt.xticks([])
    else:
        plt.xlabel('Latitude',fontsize=6)

    if (pn == 27): 
       #plt.show()
        fname = [plotpath+d1nm+"_"+d2nm+"_"+sdate+".png"]
        outname = "".join(fname)
        plt.savefig(outname)
        plt.close()

def read_datm_var(inputs, v):
    datas = np.array([])
    latss = np.array([])
    lonss = np.array([])

    datanc = nc.Dataset(inputs)
    latss = datanc.variables['lat'][:]
    lonss = datanc.variables['lon'][:]
    datas = datanc.variables[v][:,:]
    datanc.close()

    return datas, lonss, latss


def gen_figure(input1,input2,sdate,d1nm,d2nm,d1ad,d2ad,plotpath):
    dset1 = nc.Dataset(input1)
    varlists = list(dset1.variables)[3:]
    ne = 0
    for v in varlists:
        print (v)
        data1, lons1, lats1 = read_datm_var(input1, v)
        data2, lons2, lats2 = read_datm_var(input2, v)

        data1zave=np.average(data1,axis=2)
        data1zmax=np.amax(data1,axis=2)
        data1zmin=np.amin(data1,axis=2)

        data2zave=np.average(data2,axis=2)
        data2zmax=np.amax(data2,axis=2)
        data2zmin=np.amin(data2,axis=2)
        plot_zonal_ave(data1zave,data1zmax,data1zmin,lats1,data2zave,data2zmax,data2zmin,lats2,plotpath,v,sdate,d1nm,d2nm,ne)
        ne = ne + 1

if __name__ == "__main__":
   inputs = open("forcing_datm_sanity.yaml", 'r')
   #-- availe in PyYAML > 5.1
   ind = yaml.load(inputs, Loader=yaml.FullLoader)

   ym = ind["ym"]
   sym = str(ym)
   dh = ind["dh"]
   sdh = str(dh)
   sdate = sym+sdh
   d1nm = ind["d1nm"]
   d2nm = ind["d2nm"]
   d1ad = ind["d1ad"]
   d2ad = ind["d2ad"]
   input1 = ind["indir1"]+ind["d1nm"]+"/"+sym+"/"+d1ad+"."+sdate+".nc"
   input2 = ind["indir2"]+ind["d2nm"]+"/"+sym+"/"+d2ad+"."+sdate+".nc"
   plotpath = ind["plotpath"] 
 
   gen_figure(input1,input2,sdate,d1nm,d2nm,d1ad,d2ad,plotpath)


