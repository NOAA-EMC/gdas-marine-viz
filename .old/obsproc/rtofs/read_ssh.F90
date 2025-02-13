      program read_ssh
        
      integer   n_lvl
      integer   n_read
      integer   vrsn

      real,     allocatable :: ob_age (:)
      integer,  allocatable :: ob_cyc (:)
      character,allocatable :: ob_dtg (:) * 14
      real,     allocatable :: ob_lat (:)
      real,     allocatable :: ob_lon (:)
      real,     allocatable :: ob_qc (:)
      integer,  allocatable :: ob_ltc (:)
      character,allocatable :: ob_rcpt (:) * 14
      integer,  allocatable :: ob_sat (:)
      integer,  allocatable :: ob_smpl (:)
      real,     allocatable :: ob_ssh (:)
      integer,  allocatable :: ob_trck (:) 
        
      character winstart *12, winend *12

      open(10,file='ssh.bin',form='unformatted')

      write(*,*) 'reading SST'

      read (10) n_read, n_lvl, vrsn

      if (n_read .gt. 0) then

      allocate (ob_age(n_read))
      allocate (ob_cyc(n_read))
      allocate (ob_lat(n_read))
      allocate (ob_lon(n_read))
      allocate (ob_qc(n_read))
      allocate (ob_sat(n_read))
      allocate (ob_smpl(n_read))
      allocate (ob_ssh(n_read))
      allocate (ob_trck(n_read))
      allocate (ob_ltc(n_read))
      allocate (ob_dtg(n_read))
      allocate (ob_rcpt(n_read))

      read (10) ob_age(1:n_read)
      read (10) ob_cyc(1:n_read)
      read (10) ob_lat(1:n_read)
      read (10) ob_lon(1:n_read)
      read (10) ob_qc(1:n_read)
      read (10) ob_sat(1:n_read)
      read (10) ob_smpl(1:n_read)
      read (10) ob_ssh(1:n_read)
      read (10) ob_trck(1:n_read)
      read (10) ob_ltc(1:n_read)
      read (10) ob_dtg(1:n_read)
      read (10) ob_rcpt(1:n_read)

      open(21,file='window.txt',status='old')
      read(21,'(a12)') winstart
      read(21,'(a12)') winend
      close(21)

      open(20,file='ssh.txt',status='unknown')
        
      do i=1, n_read

!     HAT10
      if(ob_lat(i).ge.1.0.and.ob_lat(i).le.50.0.and.
     6   ob_lon(i).ge.-100.0.and.ob_lon(i).le.-7.0) then

!     time window
      if(ob_dtg(i)(1:12).ge.winstart.and.ob_dtg(i)(1:12).le.
     6   winend) then

      if(ob_dtg(i)(13:14).gt."59") ob_dtg(i)(13:14)="59"
      if(ob_dtg(i)(11:12).gt."59") ob_dtg(i)(11:12)="59"

      write(20,25) ob_dtg(i)(1:12),ob_lat(i),ob_lon(i),
     6             ob_ssh(i),0.1,ob_qc(i)

      endif ! End of time window
      endif ! End of HAT10

      enddo

25    format(a12,1x,f6.3,1x,f8.3,3(1x,f6.3))
      close(20)

      endif

      stop
      end

