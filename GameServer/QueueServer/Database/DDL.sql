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
    unrank BOOL NOT NULL,
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
    DECLARE is_unranked BOOL;

    SELECT unrank, overwatch_tier, valorant_tier INTO is_unranked, @ow_id, @val_id
    FROM Player
    WHERE pid = p_pid;

    IF is_unranked THEN
        SELECT 0 AS mmr;
    ELSE
        SELECT match_group INTO ow_group FROM OverwatchTier WHERE tier_id = @ow_id;
        SELECT match_group INTO val_group FROM ValorantTier WHERE tier_id = @val_id;
        SELECT GREATEST(ow_group, val_group) AS mmr;
    END IF;
END //

DELIMITER ;


INSERT INTO OverwatchTier (tier_id, name, match_group) VALUES
(1, 'overwatch_bronze_5', 0),
(2, 'overwatch_bronze_4', 0),
(3, 'overwatch_bronze_3', 0),
(4, 'overwatch_bronze_2', 0),
(5, 'overwatch_bronze_1', 1),
(6, 'overwatch_silver_5', 1),
(7, 'overwatch_silver_4', 1),
(8, 'overwatch_silver_3', 1),
(9, 'overwatch_silver_2', 1),
(10, 'overwatch_silver_1', 2),
(11, 'overwatch_gold_5', 2),
(12, 'overwatch_gold_4', 2),
(13, 'overwatch_gold_3', 2),
(14, 'overwatch_gold_2', 3),
(15, 'overwatch_gold_1', 3),
(16, 'overwatch_platinum_5', 3),
(17, 'overwatch_platinum_4', 4),
(18, 'overwatch_platinum_3', 4),
(19, 'overwatch_platinum_2', 4),
(20, 'overwatch_platinum_1', 5),
(21, 'overwatch_diamond_5', 5),
(22, 'overwatch_diamond_4', 5),
(23, 'overwatch_diamond_3', 5),
(24, 'overwatch_diamond_2', 6),
(25, 'overwatch_diamond_1', 6),
(26, 'overwatch_master_5', 6),
(27, 'overwatch_master_4', 6),
(28, 'overwatch_master_3', 6),
(29, 'overwatch_master_2', 7),
(30, 'overwatch_master_1', 7),
(31, 'overwatch_grandmaster_5', 7),
(32, 'overwatch_grandmaster_4', 7),
(33, 'overwatch_grandmaster_3', 7),
(34, 'overwatch_grandmaster_2', 7),
(35, 'overwatch_grandmaster_1', 7),
(36, 'overwatch_champion', 7);

INSERT INTO ValorantTier (tier_id, name, match_group) VALUES
(1, 'valorant_iron_1', 0),
(2, 'valorant_iron_2', 0),
(3, 'valorant_iron_3', 0),
(4, 'valorant_bronze_1', 1),
(5, 'valorant_bronze_2', 1),
(6, 'valorant_bronze_3', 1),
(7, 'valorant_silver_1', 2),
(8, 'valorant_silver_2', 2),
(9, 'valorant_silver_3', 2),
(10, 'valorant_gold_1', 3),
(11, 'valorant_gold_2', 3),
(12, 'valorant_gold_3', 3),
(13, 'valorant_platinum_1', 4),
(14, 'valorant_platinum_2', 4),
(15, 'valorant_platinum_3', 4),
(16, 'valorant_diamond_1', 5),
(17, 'valorant_diamond_2', 5),
(18, 'valorant_diamond_3', 5),
(19, 'valorant_ascendant_1', 6),
(20, 'valorant_ascendant_2', 6),
(21, 'valorant_ascendant_3', 6),
(22, 'valorant_immortal_1', 7),
(23, 'valorant_immortal_2', 7),
(24, 'valorant_immortal_3', 7),
(25, 'valorant_radiant', 7);

