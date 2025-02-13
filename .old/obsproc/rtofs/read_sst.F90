        program read_sst

        character, allocatable:: ob_dtg(:) *12
        REAL, ALLOCATABLE:: ob_age(:),ob_bias(:),ob_err(:)
        REAL, ALLOCATABLE:: ob_flg(:),ob_lat(:),ob_lon(:),ob_qc(:)
        REAL, ALLOCATABLE:: ob_sst(:)
        INTEGER, ALLOCATABLE:: ob_typ(:),ob_wm(:)

        character winstart *12, winend *12
        integer n_read
        integer vrsn
        integer i

        open(10,file='sst.bin',form='unformatted')

        n_sst = 0
        write(*,*) 'reading SST'

        read (10) n_read, n_chn, vrsn
        
        if (n_read .gt. 0) then

        allocate (ob_age(n_read))
        allocate (ob_bias(n_read))
        allocate (ob_dtg(n_read))
        allocate (ob_err(n_read))
        allocate (ob_flg(n_read))
        allocate (ob_lat(n_read))
        allocate (ob_lon(n_read))
        allocate (ob_qc(n_read))
        allocate (ob_sst(n_read))
        allocate (ob_typ(n_read))
        allocate (ob_wm(n_read))

        read (10) ob_age(1:n_read)
        read (10) ob_bias(1:n_read)
        read (10) ob_dtg(1:n_read)
        read (10) ob_err(1:n_read)
        read (10) ob_flg(1:n_read)
        read (10) ob_lat(1:n_read)
        read (10) ob_lon(1:n_read)
        read (10) ob_qc(1:n_read)
        read (10) ob_sst(1:n_read)
        read (10) ob_typ(1:n_read)
        read (10) ob_wm(1:n_read)

        open(21,file='window.txt',status='old')
        read(21,'(a12)') winstart
        read(21,'(a12)') winend
        close(21)

        open(20,file='sst.txt',status='unknown')

        do i=1, n_read

!       HAT10
        if(ob_lat(i).ge.1.0.and.ob_lat(i).le.50.0.and.
     6     ob_lon(i).ge.-100.0.and.ob_lon(i).le.-7.0) then

!       time window
        if(ob_dtg(i).ge.winstart.and.ob_dtg(i).le.winend) then

        if(ob_dtg(i)(11:12).gt."59") ob_dtg(i)(11:12)="59"

!       Missing ob_qc
        if(ob_qc(i).ge.0.0.and.ob_qc(i).lt.10.0) then

        write(20,25) ob_dtg(i),ob_lat(i),ob_lon(i),
     6               ob_sst(i),ob_err(i),ob_qc(i)

        endif ! End of missing ob_qc
        endif ! End of time window 
        endif ! End of HAT10

        enddo

25      format(a12,1x,f6.3,1x,f8.3,3(1x,f6.3))
        close(20)

        endif

        stop
        end

