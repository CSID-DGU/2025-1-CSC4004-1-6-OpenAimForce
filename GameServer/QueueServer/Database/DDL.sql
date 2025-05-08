CREATE TABLE AccountType (
    type ENUM('default', 'admin', 'club', 'collaborator') PRIMARY KEY
);

CREATE TABLE Player (
    pid INT PRIMARY KEY NOT NULL,
    ingame_id VARCHAR(64) NOT NULL UNIQUE,
    student_id INT,
    account_type ENUM('default', 'admin', 'club', 'collaborator') NOT NULL,
    contact VARCHAR(128) NOT NULL,
    mmr INT NOT NULL
);

CREATE TABLE Game (
    game_id INT PRIMARY KEY AUTO_INCREMENT,
    winner ENUM('team1', 'team2', 'draw') NOT NULL,
    game_time DATETIME NOT NULL
);

CREATE TABLE GameParticipation (
    game_id INT,
    pid INT,
    team ENUM('team1', 'team2') NOT NULL,
    kills INT NOT NULL,
    deaths INT NOT NULL,
    PRIMARY KEY (game_id, pid),
    FOREIGN KEY (game_id) REFERENCES Game(game_id) ON DELETE CASCADE,
    FOREIGN KEY (pid) REFERENCES Player(pid) ON DELETE CASCADE
);
