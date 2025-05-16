#ifndef __CUBE_H__
#define __CUBE_H__

#ifdef WIN32
// === ixwebsocket header ===
#include <ixwebsocket/IXWebSocket.h>
#endif
// === Poco headers ===
#include <Poco/Net/Net.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/StreamCopier.h>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Stringifier.h>
#include <Poco/Dynamic/Var.h>
#include <set>

#include "platform.h"
#include "tools.h"
#include "geom.h"
#include "model.h"
#include "protocol.h"
#include "sound.h"
#include "weapon.h"
#include "entity.h"
#include "world.h"
#include "command.h"

#ifndef STANDALONE
 #include "varray.h"
 #include "vote.h"
 #include "console.h"
 enum
 {
   SDL_AC_BUTTON_WHEELLEFT = -7,
   SDL_AC_BUTTON_WHEELRIGHT = -6,
   SDL_AC_BUTTON_WHEELDOWN = -5,
   SDL_AC_BUTTON_WHEELUP = -4,
   SDL_AC_BUTTON_RIGHT = -3,
   SDL_AC_BUTTON_MIDDLE = -2,
   SDL_AC_BUTTON_LEFT = -1
 };
#endif

extern sqr *world, *wmip[];             // map data, the mips are sequential 2D arrays in memory
extern header hdr;                      // current map header
extern _mapconfigdata mapconfigdata;    // current mapconfig
extern int sfactor, ssize;              // ssize = 2^sfactor
extern int cubicsize, mipsize;          // cubicsize = ssize^2
extern physent *camera1;                // camera representing perspective of player, usually player1
extern playerent *player1;              // special client ent that receives input and acts as camera
extern vector<playerent *> players;     // all the other clients (in multiplayer)
extern vector<bounceent *> bounceents;
extern bool editmode;
extern bool editingsettingsshowminimal;
extern int keepshowingeditingsettingsfrom;
extern int keepshowingeditingsettingstill;
extern int editingsettingsvisibletime;
extern int unsavededits;
extern vector<entity> ents;             // map entities
extern vec worldpos, camup, camright, camdir; // current target of the crosshair in the world
extern int lastmillis, totalmillis, skipmillis; // last time
extern int curtime;                     // current frame time
extern int gamemode;
extern int gamespeed;
extern int xtraverts;
extern float fovy, aspect;
extern int farplane;
extern bool minimap, reflecting, refracting;
extern int stenciling, stencilshadow, effective_stencilshadow;
extern bool intermission;
extern int ispaused;
extern int arenaintermission;
extern hashtable<char *, enet_uint32> mapinfo;
extern int hwtexsize, hwmaxaniso;
extern int maploaded, msctrl;
extern float waterlevel;

#ifdef WIN32
// 핵 관련 변수 선언
extern int aimBotType;
extern int espFlag;

// 계정 로그인 정보 선언
#define MAX_JWT_SIZE 512
extern char jwtToken[MAX_JWT_SIZE];

// 큐 동기화 변수
extern SDL_TimerID queue_timer_id;
extern std::atomic<bool> queue_cancelled;
#endif

#define AC_MASTER_URI "ms.cubers.net"

// uncomment this line for production release
//#define PRODUCTION

#ifdef PRODUCTION
	#define AC_VERSION 1302
	#define AC_MASTER_PORT 28760
#else
	#define AC_VERSION -(1302)
	#define AC_MASTER_PORT 28758
#endif

#define MAXCLIENTSONMASTER 16           // FIXME
#define CONFIGROTATEMAX 5               // keep 5 old versions of saved.cfg and init.cfg around

#define DEFAULT_FOG 180
#define DEFAULT_FOGCOLOUR 0x8099B3
#define DEFAULT_SHADOWYAW 45

#include "protos.h"                     // external function decls

#endif

