package quic

import (
	"log"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/lucas-clemente/quic-go/ackhandler"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

//var SMART_SCHEDULER_UPDATE_INTERVAL = time.Duration.Milliseconds(2000) #0.5
var SMART_SCHEDULER_UPDATE_INTERVAL = time.Duration(0.05 * float64(time.Second)).Milliseconds()
var RLACTIONACTIVE = 0

type time_keep struct {
	mu  sync.Mutex
	val time.Time
}

var LASTSCHEDULED *time_keep

func init() {
	// use package init to make sure path is always instantiated
	LASTSCHEDULED = new(time_keep)
	LASTSCHEDULED.val = time.Now()
}

//var publisher *ZPublisher

type scheduler struct {
	mu            sync.Mutex
	pathScheduler func(s *session) (bool, error)
	// XXX Currently round-robin based, inspired from MPTCP scheduler
	quotas         map[protocol.PathID]uint
	zclient        *ZClient
	lastScheduled  time.Time
	rlAction       chan int
	rlFailedChan   chan int
	rlFailed       bool
	rlactionActive int
	maxAction      int
	s1             rand.Source
	r1             *rand.Rand
}

func openLogFile(path string) (*os.File, error) {
	logFile, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	return logFile, nil
}

func (sch *scheduler) setup(s *session) {
	sch.quotas = make(map[protocol.PathID]uint)
	if !s.config.IgnoreRLScheduler {
		sch.scheduleToMultiplePaths()
	}
	sch.lastScheduled = time.Now()
	sch.rlAction = make(chan int, 50)
	sch.rlFailedChan = make(chan int, 50)
	sch.maxAction = 5
	sch.s1 = rand.NewSource(time.Now().UnixNano())
	sch.r1 = rand.New(sch.s1)
	go func() {
		sch.rlAction <- 0
	}()

	utils.Infof("scheduler started")

	file, err := openLogFile("./scheduler.log")
	if err != nil {
		log.Fatal(err)
	}
	log.SetOutput(file)
	log.SetFlags(log.LstdFlags | log.Lshortfile | log.Lmicroseconds)

	//log.Println("log file created")
	log.Println("Starting scheduling")
	log.Printf("IgnoreRLScheduler: %d\n", s.config.IgnoreRLScheduler)
	//log.Printf("SMART_SCHEDULER_UPDATE_INTERVAL: %d\n", SMART_SCHEDULER_UPDATE_INTERVAL)
}

//Pijus
//Set up middleware connection
func (sch *scheduler) scheduleToMultiplePaths() {
	if sch.zclient == nil {
		sch.zclient = NewClient()
		sch.zclient.Connect("ipc:///tmp/zmq")
	}
	//publisher = NewPublisher()
	//publisher.Connect("ipc:///tmp/pubsub")
	//defer publisher.Close()
}

func (sch *scheduler) getRetransmission(s *session) (hasRetransmission bool, retransmitPacket *ackhandler.Packet, pth *path) {
	// check for retransmissions first
	for {
		// TODO add ability to reinject on another path
		// XXX We need to check on ALL paths if any packet should be first retransmitted
		s.pathsLock.RLock()
	retransmitLoop:
		for _, pthTmp := range s.paths {
			retransmitPacket = pthTmp.sentPacketHandler.DequeuePacketForRetransmission()
			if retransmitPacket != nil {
				pth = pthTmp
				break retransmitLoop
			}
		}
		s.pathsLock.RUnlock()
		if retransmitPacket == nil {
			break
		}
		hasRetransmission = true

		if retransmitPacket.EncryptionLevel != protocol.EncryptionForwardSecure {
			if s.handshakeComplete {
				// Don't retransmit handshake packets when the handshake is complete
				continue
			}
			utils.Debugf("\tDequeueing handshake retransmission for packet 0x%x", retransmitPacket.PacketNumber)
			return
		}
		utils.Debugf("\tDequeueing retransmission of packet 0x%x from path %d", retransmitPacket.PacketNumber, pth.pathID)
		// resend the frames that were in the packet
		for _, frame := range retransmitPacket.GetFramesForRetransmission() {
			switch f := frame.(type) {
			case *wire.StreamFrame:
				s.streamFramer.AddFrameForRetransmission(f)
			case *wire.WindowUpdateFrame:
				// only retransmit WindowUpdates if the stream is not yet closed and the we haven't sent another WindowUpdate with a higher ByteOffset for the stream
				// XXX Should it be adapted to multiple paths?
				currentOffset, err := s.flowControlManager.GetReceiveWindow(f.StreamID)
				if err == nil && f.ByteOffset >= currentOffset {
					s.packer.QueueControlFrame(f, pth)
				}
			case *wire.PathsFrame:
				// Schedule a new PATHS frame to send
				s.schedulePathsFrame()
			default:
				s.packer.QueueControlFrame(frame, pth)
			}
		}
	}
	return
}

func (sch *scheduler) selectPathRoundRobin(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	if sch.quotas == nil {
		sch.setup(s)
	}

	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// TODO cope with decreasing number of paths (needed?)
	var selectedPath *path
	var lowerQuota, currentQuota uint
	var ok bool

	// Max possible value for lowerQuota at the beginning
	lowerQuota = ^uint(0)

pathLoop:
	for pathID, pth := range s.paths {
		// Don't block path usage if we retransmit, even on another path
		if !hasRetransmission && !pth.SendingAllowed() {
			continue pathLoop
		}

		// If this path is potentially failed, do no consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentQuota, ok = sch.quotas[pathID]
		if !ok {
			sch.quotas[pathID] = 0
			currentQuota = 0
		}

		if currentQuota < lowerQuota {
			selectedPath = pth
			lowerQuota = currentQuota
		}
	}

	return selectedPath

}

func (sch *scheduler) selectPathLowLatency(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	// FIXME Only works at the beginning... Cope with new paths during the connection
	if hasRetransmission && hasStreamRetransmission && fromPth.rttStats.SmoothedRTT() == 0 {
		// Is there any other path with a lower number of packet sent?
		currentQuota := sch.quotas[fromPth.pathID]
		for pathID, pth := range s.paths {
			if pathID == protocol.InitialPathID || pathID == fromPth.pathID {
				continue
			}
			// The congestion window was checked when duplicating the packet
			if sch.quotas[pathID] < currentQuota {
				return pth
			}
		}
	}

	var selectedPath *path
	var lowerRTT time.Duration
	var currentRTT time.Duration
	selectedPathID := protocol.PathID(255)

pathLoop:
	for pathID, pth := range s.paths {
		// Don't block path usage if we retransmit, even on another path
		if !hasRetransmission && !pth.SendingAllowed() {
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}

		currentRTT = pth.rttStats.SmoothedRTT()

		// Prefer staying single-path if not blocked by current path
		// Don't consider this sample if the smoothed RTT is 0
		if lowerRTT != 0 && currentRTT == 0 {
			continue pathLoop
		}

		// Case if we have multiple paths unprobed
		if currentRTT == 0 {
			currentQuota, ok := sch.quotas[pathID]
			if !ok {
				sch.quotas[pathID] = 0
				currentQuota = 0
			}
			lowerQuota, _ := sch.quotas[selectedPathID]
			if selectedPath != nil && currentQuota > lowerQuota {
				continue pathLoop
			}
		}

		if currentRTT != 0 && lowerRTT != 0 && selectedPath != nil && currentRTT >= lowerRTT {
			continue pathLoop
		}

		// Update
		lowerRTT = currentRTT
		selectedPath = pth
		selectedPathID = pathID
	}

	return selectedPath
}

func (sch *scheduler) choosePathsRL(s *session) {

	var avalPaths []*path

	var pathStats []*PathStats // slice

	//s.pathsLock.Lock()
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		return
	}

	// filter unavailable paths
pathLoop:
	// This forloop seems like its returning to the start
	// Not sure I understand the point, maybe we assume zero paths failure
	for pathID, pth := range s.paths {

		if !pth.SendingAllowed() {
			utils.Infof("pth.SendingAllowed() == false")
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			utils.Infof("pth.potentiallyFailed == true")
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}
		avalPaths = append(avalPaths, pth)
	}

	if len(avalPaths) < 2 {
		utils.Infof("AVAILPATHS < 2: %d", len(avalPaths))
		//sch.rlAction <- 0
		return
	}
	//s.pathsLock.Unlock()

	// Add statistics for each Path
	for _, pth := range avalPaths {
		var id uint8 = uint8(pth.pathID)
		//1048576
		var bdw uint64 = uint64(pth.cong.BandwidthEstimate()) // / 1024 //uint64(pth.bdwStats.GetBandwidth())
		var smRTT float64 = pth.rttStats.SmoothedRTT().Seconds()
		sntPkts, sntRetrans, sntLost := pth.sentPacketHandler.GetStatistics()

		s := &PathStats{
			PathID:          id,
			Bandwidth:       bdw,
			SmoothedRTT:     smRTT,
			Packets:         sntPkts,
			Retransmissions: sntRetrans,
			Losses:          sntLost}

		pathStats = append(pathStats, s)
	}

	// for _, stats := range pathStats {
	// 	utils.Infof("Statistics: PathID: %d, Bandwidth %f SmoothedRtt: %d", stats.PathID, stats.Bandwidth, stats.SmoothedRTT)
	// }

	sort.Slice(pathStats[:], func(i, j int) bool {
		return pathStats[i].PathID < pathStats[j].PathID
	})

	start := time.Now()

	request := &Request{
		StreamID:    0,
		RequestPath: "",
		Path1:       pathStats[0],
		Path2:       pathStats[1]}

	// utils.Infof("Request %d %s %s", request.ID, request.Path1.PathID, request.Path2.PathID)
	//log.Printf("Scheduler request sending to model")

	sch.mu.Lock()
	reqErr := sch.zclient.Request(request)
	if reqErr != nil {
		utils.Errorf("Error in Request\n")
		utils.Errorf(reqErr.Error())
		sch.rlFailedChan <- 0
	}

	//log.Printf("Listening for model response")
	response, err := sch.zclient.Response()
	if err != nil {
		utils.Errorf("Error in Response\n")
		utils.Errorf(err.Error())
		sch.rlFailedChan <- 0
	}
	sch.mu.Unlock()

	//_ = response

	//streamInfo := &StreamInfo{
	//	StreamID:       0,
	//	ObjectID:       "",
	//	CompletionTime: 0,
	//	Path:           ""}
	//publisher.Publish(streamInfo)

	elapsed := time.Since(start)
	// utils.Infof("ZClient Response.ID: %d, Response.PathID: %d\n", response.ID, response.PathID)
	utils.Infof("Communication overhead %s", elapsed)
	//log.Printf("Communication overhead %s\n", elapsed)
	//-----------------------------------------------------------------------------------------

	// assign all volume to specified agent path
	//var selectedPathID protocol.PathID = protocol.PathID(response.PathID)
	/*
		var selectedPath = avalPaths[0]
		if response.PathID < uint8(len(avalPaths)) {
			selectedPath = avalPaths[selectedPathID]
			utils.Infof("Received invalid path id: %u", response.PathID)
		}
	*/
	//var selectedPath = s.paths[selectedPathID]
	sch.rlAction <- int(response.PathID)
	//selectedPaths[s.paths[selectedPathID]] = float64(stream.size)
	return
}

func (sch *scheduler) propabilisticScheduler(s *session, hasRetransmission bool, prob1 float32, prob2 float32) *path {
	var avalPaths []*path

	//s.pathsLock.Lock()
	// XXX Avoid using PathID 0 if there is more than 1 path
	if len(s.paths) <= 1 {
		if !hasRetransmission && !s.paths[protocol.InitialPathID].SendingAllowed() {
			return nil
		}
		return s.paths[protocol.InitialPathID]
	}

	var selectedPath *path
	// filter unavailable paths
pathLoop:
	// This forloop seems like its returning to the start
	// Not sure I understand the point, maybe we assume zero paths failure
	for pathID, pth := range s.paths {

		if !pth.SendingAllowed() {
			//utils.Infof("pth.SendingAllowed() == false")
			continue pathLoop
		}

		// If this path is potentially failed, do not consider it for sending
		if pth.potentiallyFailed.Get() {
			//utils.Infof("pth.potentiallyFailed == true")
			continue pathLoop
		}

		// XXX Prevent using initial pathID if multiple paths
		if pathID == protocol.InitialPathID {
			continue pathLoop
		}
		avalPaths = append(avalPaths, pth)
	}

	if len(avalPaths) < 2 {
		//utils.Infof("AVAILPATHS < 2: %d", len(avalPaths))
		//return s.paths[avalPaths[0].pathID]
		return nil
	}
	//s.pathsLock.Unlock()

	// Add statistics for each Path
	/*
		for _, pth := range avalPaths {

		}
	*/
	sort.Slice(avalPaths[:], func(i, j int) bool {
		return avalPaths[i].pathID < avalPaths[j].pathID
	})

	randv := sch.r1.Float32()
	if prob1 > randv {
		selectedPath = avalPaths[0]
	} else {
		selectedPath = avalPaths[1]
	}

	return selectedPath
}

func (sch *scheduler) selectPathSmart(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {

	select {
	case msg := <-sch.rlAction:
		//sch.rlactionActive = msg
		RLACTIONACTIVE = msg
		//log.Printf("RL action received")
		utils.Infof("RL action received: %d", msg)
	default:

	}
	select {
	case msg := <-sch.rlFailedChan:
		//sch.rlactionActive = msg
		_ = msg
		sch.rlFailed = true
		//log.Printf("RL action received")
	default:

	}

	LASTSCHEDULED.mu.Lock()
	elapsed := time.Since(LASTSCHEDULED.val).Milliseconds()
	if elapsed > SMART_SCHEDULER_UPDATE_INTERVAL && !sch.rlFailed {
		//sch.lastScheduled = time.Now()
		LASTSCHEDULED.val = time.Now()
		go sch.choosePathsRL(s)
	}
	LASTSCHEDULED.mu.Unlock()

	action := RLACTIONACTIVE
	if action == 0 {
		return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
	} else if action >= 1 && action <= sch.maxAction {
		prob1 := float32(action-1) / float32(sch.maxAction-1)
		path := sch.propabilisticScheduler(s, hasRetransmission, prob1, 1-prob1)
		if path == nil {
			return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
		} else {
			return path
		}
	} else {
		utils.Errorf("RL action not in bounds: %d", action)
	}
	return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
}

// Lock of s.paths must be held
func (sch *scheduler) selectPath(s *session, hasRetransmission bool, hasStreamRetransmission bool, fromPth *path) *path {
	// XXX Currently round-robin
	//_todo select the right scheduler dynamically
	//return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
	//log.Printf("IgnoreRLScheduler: %d\n", s.config.IgnoreRLScheduler)
	if !s.config.IgnoreRLScheduler {
		return sch.selectPathSmart(s, hasRetransmission, hasStreamRetransmission, fromPth)
	} else {
		return sch.selectPathLowLatency(s, hasRetransmission, hasStreamRetransmission, fromPth)
	}
	// return sch.selectPathRoundRobin(s, hasRetransmission, hasStreamRetransmission, fromPth)
}

// Lock of s.paths must be free (in case of log print)
func (sch *scheduler) performPacketSending(s *session, windowUpdateFrames []*wire.WindowUpdateFrame, pth *path) (*ackhandler.Packet, bool, error) {
	// add a retransmittable frame
	if pth.sentPacketHandler.ShouldSendRetransmittablePacket() {
		s.packer.QueueControlFrame(&wire.PingFrame{}, pth)
	}

	packet, err := s.packer.PackPacket(pth)

	if err != nil || packet == nil {
		return nil, false, err
	}

	if err = s.sendPackedPacket(packet, pth); err != nil {
		return nil, false, err
	}

	// send every window update twice
	for _, f := range windowUpdateFrames {
		s.packer.QueueControlFrame(f, pth)
	}

	// Packet sent, so update its quota
	sch.quotas[pth.pathID]++

	// Provide some logging if it is the last packet
	for _, frame := range packet.frames {
		switch frame := frame.(type) {
		case *wire.StreamFrame:
			if frame.FinBit {
				// Last packet to send on the stream, print stats
				s.pathsLock.RLock()
				utils.Infof("Info for stream %x of %x", frame.StreamID, s.connectionID)
				for pathID, pth := range s.paths {
					sntPkts, sntRetrans, sntLost := pth.sentPacketHandler.GetStatistics()
					rcvPkts := pth.receivedPacketHandler.GetStatistics()
					utils.Infof("Path %x: sent %d retrans %d lost %d; rcv %d rtt %v", pathID, sntPkts, sntRetrans, sntLost, rcvPkts, pth.rttStats.SmoothedRTT())
				}
				s.pathsLock.RUnlock()
			}
		default:
		}
	}

	pkt := &ackhandler.Packet{
		PacketNumber:    packet.number,
		Frames:          packet.frames,
		Length:          protocol.ByteCount(len(packet.raw)),
		EncryptionLevel: packet.encryptionLevel,
	}

	return pkt, true, nil
}

// Lock of s.paths must be free
func (sch *scheduler) ackRemainingPaths(s *session, totalWindowUpdateFrames []*wire.WindowUpdateFrame) error {
	// Either we run out of data, or CWIN of usable paths are full
	// Send ACKs on paths not yet used, if needed. Either we have no data to send and
	// it will be a pure ACK, or we will have data in it, but the CWIN should then
	// not be an issue.
	s.pathsLock.RLock()
	defer s.pathsLock.RUnlock()
	// get WindowUpdate frames
	// this call triggers the flow controller to increase the flow control windows, if necessary
	windowUpdateFrames := totalWindowUpdateFrames
	if len(windowUpdateFrames) == 0 {
		windowUpdateFrames = s.getWindowUpdateFrames(s.peerBlocked)
	}
	for _, pthTmp := range s.paths {
		ackTmp := pthTmp.GetAckFrame()
		for _, wuf := range windowUpdateFrames {
			s.packer.QueueControlFrame(wuf, pthTmp)
		}
		if ackTmp != nil || len(windowUpdateFrames) > 0 {
			if pthTmp.pathID == protocol.InitialPathID && ackTmp == nil {
				continue
			}
			swf := pthTmp.GetStopWaitingFrame(false)
			if swf != nil {
				s.packer.QueueControlFrame(swf, pthTmp)
			}
			s.packer.QueueControlFrame(ackTmp, pthTmp)
			// XXX (QDC) should we instead call PackPacket to provides WUFs?
			var packet *packedPacket
			var err error
			if ackTmp != nil {
				// Avoid internal error bug
				packet, err = s.packer.PackAckPacket(pthTmp)
			} else {
				packet, err = s.packer.PackPacket(pthTmp)
			}
			if err != nil {
				return err
			}
			err = s.sendPackedPacket(packet, pthTmp)

			if err != nil {
				return err
			}
		}
	}
	s.peerBlocked = false
	return nil
}

func (sch *scheduler) sendPacket(s *session) error {
	var pth *path

	// Update leastUnacked value of paths
	s.pathsLock.RLock()
	for _, pthTmp := range s.paths {
		pthTmp.SetLeastUnacked(pthTmp.sentPacketHandler.GetLeastUnacked())
	}
	s.pathsLock.RUnlock()

	// get WindowUpdate frames
	// this call triggers the flow controller to increase the flow control windows, if necessary
	windowUpdateFrames := s.getWindowUpdateFrames(false)
	for _, wuf := range windowUpdateFrames {
		s.packer.QueueControlFrame(wuf, pth)
	}

	// Repeatedly try sending until we don't have any more data, or run out of the congestion window
	for {
		// We first check for retransmissions
		hasRetransmission, retransmitHandshakePacket, fromPth := sch.getRetransmission(s)
		// XXX There might still be some stream frames to be retransmitted
		hasStreamRetransmission := s.streamFramer.HasFramesForRetransmission()

		// Select the path here
		s.pathsLock.RLock()
		pth = sch.selectPath(s, hasRetransmission, hasStreamRetransmission, fromPth)
		s.pathsLock.RUnlock()

		// XXX No more path available, should we have a new QUIC error message?
		if pth == nil {
			windowUpdateFrames := s.getWindowUpdateFrames(false)
			return sch.ackRemainingPaths(s, windowUpdateFrames)
		}

		// If we have an handshake packet retransmission, do it directly
		if hasRetransmission && retransmitHandshakePacket != nil {
			s.packer.QueueControlFrame(pth.sentPacketHandler.GetStopWaitingFrame(true), pth)
			packet, err := s.packer.PackHandshakeRetransmission(retransmitHandshakePacket, pth)
			if err != nil {
				return err
			}
			if err = s.sendPackedPacket(packet, pth); err != nil {
				return err
			}
			continue
		}

		// XXX Some automatic ACK generation should be done someway
		var ack *wire.AckFrame
		ack = pth.GetAckFrame()
		if ack != nil {
			s.packer.QueueControlFrame(ack, pth)
		}
		if ack != nil || hasStreamRetransmission {
			swf := pth.sentPacketHandler.GetStopWaitingFrame(hasStreamRetransmission)
			if swf != nil {
				s.packer.QueueControlFrame(swf, pth)
			}
		}

		// Also add CLOSE_PATH frames, if any
		for cpf := s.streamFramer.PopClosePathFrame(); cpf != nil; cpf = s.streamFramer.PopClosePathFrame() {
			s.packer.QueueControlFrame(cpf, pth)
		}

		// Also add ADD ADDRESS frames, if any
		for aaf := s.streamFramer.PopAddAddressFrame(); aaf != nil; aaf = s.streamFramer.PopAddAddressFrame() {
			s.packer.QueueControlFrame(aaf, pth)
		}

		// Also add PATHS frames, if any
		for pf := s.streamFramer.PopPathsFrame(); pf != nil; pf = s.streamFramer.PopPathsFrame() {
			s.packer.QueueControlFrame(pf, pth)
		}

		pkt, sent, err := sch.performPacketSending(s, windowUpdateFrames, pth)
		if err != nil {
			return err
		}
		windowUpdateFrames = nil
		if !sent {
			// Prevent sending empty packets
			return sch.ackRemainingPaths(s, windowUpdateFrames)
		}

		// Duplicate traffic when it was sent on an unknown performing path
		// FIXME adapt for new paths coming during the connection
		if pth.rttStats.SmoothedRTT() == 0 {
			currentQuota := sch.quotas[pth.pathID]
			// Was the packet duplicated on all potential paths?
		duplicateLoop:
			for pathID, tmpPth := range s.paths {
				if pathID == protocol.InitialPathID || pathID == pth.pathID {
					continue
				}
				if sch.quotas[pathID] < currentQuota && tmpPth.sentPacketHandler.SendingAllowed() {
					// Duplicate it
					pth.sentPacketHandler.DuplicatePacket(pkt)
					break duplicateLoop
				}
			}
		}

		// And try pinging on potentially failed paths
		if fromPth != nil && fromPth.potentiallyFailed.Get() {
			err = s.sendPing(fromPth)
			if err != nil {
				return err
			}
		}
	}
}
