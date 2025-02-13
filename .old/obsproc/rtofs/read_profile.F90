      program read_profile
c     implicit none
c
c     ..define maximum number daily files
c
      integer    MX_FILES
      parameter (MX_FILES = 300)
c
      integer    UNIT
      parameter (UNIT = 20)
c
      integer   n_depth
      integer   n_prf
      parameter(n_depth=2500, n_prf=7500)

c
      character winstart *12, winend *12
      logical   exist
      character file_dtg (MX_FILES) * 10
      character file_name * 256
      integer   i, j
      integer   len
      integer   len_data
      integer   mx_depth
      integer   mx_obs
      integer   n_dup
      integer   n_files
      integer   n_in
      integer   n_lev
      integer   n_out
      integer   n_rpl
      logical   new_file
      integer   old_vrsn
      real      qc_lmt
      integer   total
      integer   vrsn

      real      clm_sal (n_depth, n_prf)
      real      clm_sstd (n_depth, n_prf)
      real      clm_tmp (n_depth, n_prf)
      real      clm_tstd (n_depth, n_prf)
      logical   dup_prf (n_prf)
      real      prf_btm (n_prf)
c     character prf_dtg (n_prf) * 12
      integer   ob_flg (n_depth, n_prf)
c     real      prf_lat (n_prf)
      real      prf_lon (n_prf)
      integer   prf_ls (n_prf)
c      integer   prf_lt (n_prf)
      integer,  allocatable :: prf_lt (:)
c     real      prf_lvl (n_depth, n_prf)
      character,allocatable :: prf_dtg (:) * 12
      real,     allocatable :: prf_lat (:)
      real,     allocatable :: prf_lvl (:,:)
      character prf_rcpt (n_prf) * 12
      integer   prf_rej (n_depth, n_prf)
      real      prf_sal (n_depth, n_prf)
      real      prf_sal_err (n_depth, n_prf)
      integer   prf_sal_typ (n_prf)
      character prf_sgn (n_prf) * 7
      real      prf_sprb (n_depth, n_prf)

      character prf_csal (n_depth, n_prf) * 7
      real      prf_cssd (n_depth, n_prf)
      character prf_ctmp (n_depth, n_prf) * 7
      real      prf_ctsd (n_depth, n_prf)

      real      prf_sqc (n_prf)
      real      prf_tmp (n_depth, n_prf)
      real      prf_tmp_err (n_depth, n_prf)
      integer   prf_tmp_typ (n_prf)
      real      prf_tprb (n_depth, n_prf)
      real      prf_tqc (n_prf)
      real      prf_rct (n_prf)

      open (UNIT, file='profile.bin', status='old',
     6      form='unformatted')

      read (UNIT) n_in, n_lev, old_vrsn
      write(*,*) ' n_in, n_lev, old_vrsn',  n_in, n_lev, old_vrsn

      if (n_in .gt. 0) then
      allocate (prf_lat (n_in))
      allocate (prf_dtg (n_in))
      allocate (prf_lt (n_in))
      allocate (prf_lvl (n_lev, n_in))

      read (unit) prf_btm(1:n_in)
      read (unit) prf_lat(1:n_in)
      read (unit) prf_lon(1:n_in)
      read (unit) prf_ls(1:n_in)
      read (unit) prf_lt(1:n_in)
      read (unit) prf_sal_typ(1:n_in)
      read (unit) prf_sqc(1:n_in)
      read (unit) prf_tmp_typ(1:n_in)
      read (unit) prf_tqc(1:n_in)

      do i = 1, n_in
         read (unit) prf_lvl(1:prf_lt(i),i)
         read (unit) prf_sal(1:prf_lt(i),i)
         read (unit) prf_sal_err(1:prf_lt(i),i)
         read (unit) prf_sprb(1:prf_lt(i),i)
         read (unit) prf_tmp(1:prf_lt(i),i)
         read (unit) prf_tmp_err(1:prf_lt(i),i)
         read (unit) prf_tprb(1:prf_lt(i),i)
         read (UNIT) ob_clm_sal
         read (unit) prf_cssd(1:prf_lt(i),i)
         read (UNIT) ob_clm_tmp
         read (unit) prf_ctsd(1:prf_lt(i),i)
         read (unit) ob_flg(1:prf_lt(i),i)
      enddo

      read (unit) prf_dtg(1:n_in)
      read (unit) prf_rct(1:n_in)
      read (unit) prf_sgn(1:n_in)

      close (unit)

      endif

      open(21,file='window.txt',status='old')
      read(21,'(a12)') winstart
      read(21,'(a12)') winend
      close(21)

      open(20,file='profile.txt',status='unknown')

      do i=1, n_in

!     HAT10
      if(prf_lat(i).ge.1.0.and.prf_lat(i).le.50.0.and.
     6   prf_lon(i).ge.-100.0.and.prf_lon(i).le.-7.0) then

!     time window
      if(prf_dtg(i).ge.winstart.and.prf_dtg(i).le.winend) then

      if(prf_tqc(i).gt.99.0) then
         prf_tqc(i)=99.9
      endif

      if(prf_sqc(i).gt.99.0) then
         prf_sqc(i)=99.9
      endif

!     profile
      do j=1, prf_lt(i)

      if(prf_sal_err(j,i).eq.-999.0) then
      prf_sal_err(j,i)=1.0
      endif

      write(20,25) prf_dtg(i),prf_lat(i),prf_lon(i),
     6             prf_tmp(j,i),prf_tmp_err(j,i),prf_tqc(i),
     6             prf_sal(j,i),prf_sal_err(j,i),prf_sqc(i),
     6             prf_lvl(j,i)

      enddo ! profile

      endif ! time window

      endif ! HAT10

      enddo

25    format(a12,1x,f6.3,1x,f8.3,6(1x,f6.3),1x,f6.1)
      close(20)

      stop
      end
