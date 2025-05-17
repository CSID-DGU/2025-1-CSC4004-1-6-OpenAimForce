package bridge

import "sync"

var sendFunc func(string)
var mu sync.RWMutex

func RegisterSender(fn func(string)) {
	mu.Lock()
	sendFunc = fn
	mu.Unlock()
}

func SendToMaster(msg string) {
	mu.RLock()
	defer mu.RUnlock()
	if sendFunc != nil {
		sendFunc(msg)
	}
}
