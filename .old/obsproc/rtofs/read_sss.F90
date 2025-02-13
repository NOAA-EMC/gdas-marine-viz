      program read_sss

      real,     allocatable :: ob_age (:)
      character,allocatable :: ob_dtg (:) * 12
      real,     allocatable :: ob_err (:)
      integer,  allocatable :: ob_flg (:)
      real,     allocatable :: ob_lat (:)
      real,     allocatable :: ob_lon (:)
      real,     allocatable :: ob_qc (:)
      character,allocatable :: ob_rcp (:) * 12
      real,     allocatable :: ob_sss (:)
      real,     allocatable :: ob_sst (:)
      integer,  allocatable :: ob_typ (:)
c
      character winstart *12, winend *12
      integer n_read, n_lvl, vrsn, i
c
      open(10,file='sss.bin',form='unformatted')

      write(*,*) 'reading RTOFS binary SSS'

      read (10) n_read, n_lvl, vrsn
      if (n_read .gt. 0) then
c
      allocate (ob_age (n_read))
      allocate (ob_dtg (n_read))
      allocate (ob_err (n_read))
      allocate (ob_flg (n_read))
      allocate (ob_lat (n_read))
      allocate (ob_lon (n_read))
      allocate (ob_qc (n_read))
      allocate (ob_rcp (n_read))
      allocate (ob_sss (n_read))
      allocate (ob_sst (n_read))
      allocate (ob_typ (n_read))

      read (10) ob_age(1:n_read)
      read (10) ob_err(1:n_read)
      read (10) ob_flg(1:n_read)
      read (10) ob_lat(1:n_read)
      read (10) ob_lon(1:n_read)
      read (10) ob_qc(1:n_read)
      read (10) ob_typ(1:n_read)
      read (10) ob_sss(1:n_read)
      read (10) ob_sst(1:n_read)
      read (10) ob_dtg(1:n_read)

      if (vrsn .eq. 2) then
         read (10) ob_rcp(1:n_read)
      else
         do i = 1, n_read
         ob_rcp(i) = ob_dtg(i)
         enddo
      endif

      open(21,file='window.txt',status='old')
      read(21,'(a12)') winstart
      read(21,'(a12)') winend
      close(21)

      open(20,file='sss.txt',status='new')

      do i=1, n_read

!     HAT10
      if(ob_lat(i).ge.1.0.and.ob_lat(i).le.50.0.and.
     6     ob_lon(i).ge.-100.0.and.ob_lon(i).le.-7.0) then

!     time window
      if(ob_dtg(i).ge.winstart.and.ob_dtg(i).le.winend) then

      if(ob_dtg(i)(11:12).gt."59") ob_dtg(i)(11:12)="59"

!     Missing ob_qc
      if(ob_qc(i).gt.99.0) ob_qc(i)=99.9
      if(ob_err(i).gt.9.9) ob_err(i)=9.9

      write(20,25) ob_age(i),ob_dtg(i),ob_err(i),ob_lat(i),
     6             ob_lon(i),ob_qc(i),ob_sss(i)

      endif ! time window 

      endif ! HAT10

      enddo

25    format(f8.1,1x,a12,1x,f4.2,1x,f6.3,1x,f8.3,1x,f6.3,1x,f6.3)
      close(20)

      endif

      stop
      end


