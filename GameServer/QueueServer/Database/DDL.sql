CREATE TABLE AccountType (
    type ENUM('default', 'admin', 'club', 'collaborator') PRIMARY KEY
);

CREATE TABLE OverwatchTier (
    tier_id INT PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE,
    match_group INT NOT NULL DEFAULT 0
);

CREATE TABLE ValorantTier (
    tier_id INT PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE,
    match_group INT NOT NULL DEFAULT 0
);

CREATE TABLE Player (
    pid INT PRIMARY KEY AUTO_INCREMENT NOT NULL,
    account_type ENUM('default', 'admin', 'club', 'collaborator') NOT NULL,
    ingame_id VARCHAR(64) NOT NULL UNIQUE,
    student_id INT,
    password VARCHAR(255) NOT NULL,
    real_name VARCHAR(64) NOT NULL,
    contact VARCHAR(128) NOT NULL,
    overwatch_tier INT,
    valorant_tier INT,
    etc_tier VARCHAR(255),
    FOREIGN KEY (overwatch_tier) REFERENCES OverwatchTier(tier_id),
    FOREIGN KEY (valorant_tier) REFERENCES ValorantTier(tier_id)
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

DELIMITER //

CREATE PROCEDURE get_mmr(IN p_pid INT)
BEGIN
    DECLARE ow_group INT DEFAULT 0;
    DECLARE val_group INT DEFAULT 0;

    SELECT overwatch_tier, valorant_tier INTO @ow_id, @val_id
    FROM Player
    WHERE pid = p_pid;

    SELECT match_group INTO ow_group FROM OverwatchTier WHERE tier_id = @ow_id;
    SELECT match_group INTO val_group FROM ValorantTier WHERE tier_id = @val_id;
    SELECT GREATEST(ow_group, val_group) AS mmr;
END //

DELIMITER ;


INSERT INTO OverwatchTier (tier_id, name, match_group) VALUES
(0, 'overwatch_unrank', 0),
(1, 'overwatch_bronze', 0),
(2, 'overwatch_silver', 0),
(3, 'overwatch_gold', 0),
(4, 'overwatch_platinum', 0),
(5, 'overwatch_diamond', 1),
(6, 'overwatch_master', 1),
(7, 'overwatch_grandmaster', 1),
(8, 'overwatch_champion', 1);

INSERT INTO ValorantTier (tier_id, name, match_group) VALUES
(0, 'valorant_unrank', 0),
(1, 'valorant_iron', 0),
(2, 'valorant_bronze', 0),
(3, 'valorant_silver', 0),
(4, 'valorant_gold', 1),
(5, 'valorant_platinum', 1),
(6, 'valorant_diamond', 1),
(7, 'valorant_ascendant', 2),
(8, 'valorant_immortal', 2),
(9, 'valorant_radiant', 2);

