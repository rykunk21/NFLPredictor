"""
@File: Data.py
@Author: Ryan Kunkel
@AIAssistant: GPT-4

"""

from typing import Any
import os

class Manager:
    """
    Populates Datasets
    """
    def __init__(self, year) -> None:
        self.year = year

    ### PUBLIC

    def pullRecaps(self) -> None:
        """
        Pulls the game recaps
        """
        # Iterate over the xlsx files
        
        file_path = os.path.join(RECAPS_DIR, f"{self.year}.xlsx")
        workbook = openpyxl.load_workbook(file_path)
        
        # Keeping a set to check for repeated gameIDs
        seen_game_ids = set()
        
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows = list(sheet.iter_rows(min_row=2, max_col=2, values_only=True))
            
            i = 0
            while i < len(rows):
                game_id, teamname = rows[i]
                
                # Check if gameID is not repeated
                if game_id not in seen_game_ids:
                    seen_game_ids.add(game_id)
                    
                    url = f"https://www.espn.com/nfl/recap/_/gameId/{game_id}"
                    try:
                        with urllib.request.urlopen(url) as response:
                            content = response.read()
                            soup = BeautifulSoup(content, 'html.parser')
                            main_div = soup.find("div", {"role": "main"})
                            
                            if main_div:
                                article_text = " ".join(main_div.stripped_strings)
                                
                                # team2 name will be in the next row
                                team2name = rows[i + 1][1]
                                
                                # Create necessary directories
                                year_dir = os.path.join(self.recapDir, str(self.year))
                                week_dir = os.path.join(year_dir, sheet_name)
                                os.makedirs(week_dir, exist_ok=True)
                                
                                # Save the article
                                recap_file_path = os.path.join(week_dir, f"{teamname}vs{team2name}.txt")
                                with open(recap_file_path, 'w', encoding='utf-8') as file:
                                    file.write(article_text)
                    except urllib.error.HTTPError as err:
                        print(f'ERROR FROM {teamname}, {self.year}')
                        
                # Skip the next row as it's the second team of the same game
                i += 2

    def pullScores(self) -> None:
        def week():
            for x in range(1, 5):
                yield (x, 1)
            for x in range(1, 19):
                yield (x, 2)
            for x in range(1, 6):
                yield (x, 3)
        
        for weeknum, seasontype in week():
            if seasontype == 1:
                sheetName = f'P{weeknum}'
            elif seasontype == 2:
                sheetName = f'W{weeknum}'
            else:
                sheetName = f'PO{weeknum}'


            gameData = self._weeklyScores(self._weekURL(weeknum, seasontype, self.year))
            self._write_to_xlsx(gameData, sheetName, file_name=f'{self.year}.xlsx')

    def updateElos(self) -> None:

        simluator = Simluator(self.year)
        for i in range(1, CURRENT_WEEK + 1):
            simluator.simulate_week(i)

    ### PRIVATE

    def _weekURL(self, gameNum, seasonType, year):
        return f'https://www.espn.com/nfl/scoreboard/_/week/{gameNum}/year/{year}/seasontype/{seasonType}'

    def _weeklyScores(self, url = None):
        response = urllib.request.urlopen(url)

        # Check the response status
        if 200 <= response.getcode() < 300:
            # Read the content and create a soup object
            content = response.read()
            soup = BeautifulSoup(content, 'html.parser')
            
            
            ## Extract the games section by class
            module = soup.find('section', class_='gameModules')

            # Initialize an empty list to hold the siblings
            modules = [module]

            # Iterate over the siblings of the 'modules' tag
            for sibling in module.next_siblings:
                if sibling.name:  # This check is to filter out NavigableString objects, which may not be actual tags
                    modules.append(sibling)

            ids = dict()
            # Iterate over the siblings of the 'modules' tag
            for element in modules:
                if element.name == 'section':
                    # Find the direct child div
                    div_child = element.find('div', recursive=False)
                    if div_child:
                        # Iterate over section tags inside the div
                        for nested_section in div_child.find_all('section', recursive=False):
                            teamlinks = nested_section.find_all('a', class_='truncate')
                            teamlinks = list(map(lambda x: x['href'].split('/')[-1], teamlinks))
                            teamlinks = list(map(lambda x: x.split('-')[-1], teamlinks))

                            if not teamlinks:
                                return

                            li1, li2 = nested_section.ul.find_all('li', recursive=False)
                            try:
                                t1scores = []
                                for child in li1.find_all('div', recursive = False)[1]:
                                    t1scores.append(int(child.text))
                                
                                t2scores = []
                                for child in li2.find_all('div', recursive = False)[1]:
                                    t2scores.append(int(child.text))

                                # Regular game with no OT
                                if len(t1scores) == 4 and len(t2scores) == 4:
                                    t1scores.append(0)
                                    t2scores.append(0) 

                                    t1scores.append(sum(t1scores))
                                    t2scores.append(sum(t2scores)) 

                                # game went into OT
                                elif len(t1scores) == 5 and len(t2scores) == 5:

                                    t1scores.append(sum(t1scores))
                                    t2scores.append(sum(t2scores))

                                section_id = nested_section.get('id')
                                if section_id:  # Check if the section has an ID attribute
                                    ids[section_id] = ({teamlinks[0]: tuple(t1scores)}, {teamlinks[1]: tuple(t2scores)})
                            except:
                                section_id = nested_section.get('id')
                                if section_id:  # Check if the section has an ID attribute
                                    ids[section_id] = ({teamlinks[0]: (0,0,0,0,0)}, {teamlinks[1]: (0,0,0,0,0)})

            # Print out the result
            return ids


        else:
            raise ValueError(f"Failed to retrieve content from {url}. HTTP Response Code: {response.getcode()}")

    def _write_to_xlsx(self, data_dict, sheet_name, file_name):

        file_name = os.path.join(SCORES_DIR, file_name)
        # Check if the workbook already exists
        if os.path.exists(file_name):
            wb = openpyxl.load_workbook(file_name)
            # Check if sheet_name already exists; if so, remove it to avoid duplication
            if sheet_name in wb.sheetnames:
                del wb[sheet_name]
            ws = wb.create_sheet(title=sheet_name)
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = sheet_name

        # Write the headers
        headers = ['ID', 'teamname', 'q1', 'q2', 'q3', 'q4', 'OT', 'finalscore', 'home', 'win']
        ws.append(headers)

        # handle null data: Write nonthing:
        if not data_dict:
            wb.save(file_name)
            return

        # Iterate over the data dictionary and extract data
        for game_id, teams in data_dict.items():
            away_team, away_scores = list(teams[0].items())[0]
            home_team, home_scores = list(teams[1].items())[0]

            if not away_scores or not home_scores:
                wb.save(file_name)
                print(f'Saved {self.year}: {sheet_name}')
                return
            
            away_win = 1 if away_scores[-2] > home_scores[-2] else 0  # Adjusted index due to OT addition
            home_win = 1 - away_win

            # Handle games without OT
            if len(away_scores) == 6 and len(home_scores) == 6:
                # Write away team data
                ws.append([game_id, away_team] + list(away_scores[:-1]) + [away_scores[-1], 0, away_win])
                # Write home team data
                ws.append([game_id, home_team] + list(home_scores[:-1]) + [home_scores[-1], 1, home_win])
            else:
                # Write away team data
                ws.append([game_id, away_team] + list(away_scores) + [0, away_win])
                # Write home team data
                ws.append([game_id, home_team] + list(home_scores) + [1, home_win])


        # Save the workbook
        wb.save(file_name)
        print(f'Saved {self.year}: {sheet_name}')
    

class Season:
    """
    On demand season information
    """
    def __init__(self):
        pass
    
    def game(self, game):
        pass


class Game:

    def __init__(self, *args, **kwargs) -> None:
        
        week = kwargs.get('week')
        year = kwargs.get('year')
        

        if week and year:
            team = kwargs.get('team')

            # Search the datasets for the game ID
            file_name = os.path.join(SCORES_DIR, f'{year}.xlsx')

            # Check if the workbook already exists
            if os.path.exists(file_name):
                wb = openpyxl.load_workbook(file_name)
                if week in wb.sheetnames:
                    ws = wb[week]
                    
                    # Get column names to indices mapping
                    col_names = {cell.value: idx for idx, cell in enumerate(ws[1])}  # Assuming first row contains column names

                    for row in ws.iter_rows(min_row=2):  # Start from the second row to skip the header
                        current_team_name = row[col_names['teamname']].value
                        elo = row[col_names['elo']].value

                        if current_team_name == team:
                            self.ID = row[col_names['ID']].value

                            # Create Team instance
                            initParams = {
                                'name': current_team_name,
                                'elo': elo,
                                'home': row[col_names['home']].value == 1
                            }
                            if initParams['home']:
                                homeTeamObj = Team(**initParams)
                                homeTeamFinalScore = row[col_names['finalscore']].value  # Assuming you have a 'finalscore' column

                            else:
                                awayTeamObj = Team(**initParams)
                                awayTeamFinalScore = row[col_names['finalscore']].value  # Assuming you have a 'finalscore' column

                            break

                    # Now, find the other team with the same ID
                    for row in ws.iter_rows(min_row=2):  # Start from the second row to skip the header
                        current_team_name = row[col_names['teamname']].value
                        if row[col_names['ID']].value == self.ID and current_team_name != team:
                            elo = row[col_names['elo']].value

                            # Create Team instance
                            initParams = {
                                'name': current_team_name,
                                'elo': elo,
                                'home': row[col_names['home']].value == 1
                            }
                            if initParams['home']:
                                homeTeamObj = Team(**initParams)
                                homeTeamFinalScore = row[col_names['finalscore']].value  # Assuming you have a 'finalscore' column

                            else:
                                awayTeamObj = Team(**initParams)
                                awayTeamFinalScore = row[col_names['finalscore']].value  # Assuming you have a 'finalscore' column

                            break

                    self.home = homeTeamObj
                    self.away = awayTeamObj

                    self.scoreboard = Scoreboard(self.home, self.away, homeTeamFinalScore, awayTeamFinalScore)

                else:
                    raise Exception(f"Week '{week}' does not exist in the workbook.")
            else:
                raise Exception('Game year does not exist')
            
    def sumamry(self):
        url = f"https://www.espn.com/nfl/recap/_/gameId/{self.ID}"

        try:
            with urllib.request.urlopen(url) as response:
                content = response.read()
                soup = BeautifulSoup(content, 'html.parser')
                main_div = soup.find("div", {"role": "main"})
                
                if main_div:
                    return " ".join(main_div.stripped_strings)
                    
                
        except urllib.error.HTTPError as err:
            print(f'ERROR FROM RETRIEVING GAME FROM, {self.year}')

    def probabilityPlot(self, visual=False): #TODO
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.action_chains import ActionChains


        url = f'https://www.espn.com/nfl/boxscore/_/gameId/{self.ID}'

        options = webdriver.ChromeOptions()
        if not visual:
            options.add_argument('--headless')

        with webdriver.Chrome(options=options) as driver:
            driver.get(url)

            action = ActionChains(driver)
            
            xPath = '//*[@id="themeProvider"]/div/div/div[5]/div/div[1]/div[2]/section/div'
            chart = driver.find_element_by_xpath(xPath)

    def playByPlay(self, visual=False): #TODO
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from time import sleep

        url = f'https://www.espn.com/nfl/playbyplay/_/gameId/{self.ID}'

        options = webdriver.ChromeOptions()
        if not visual:
            options.add_argument('--headless')

        plays = []



        with webdriver.Chrome(options=options) as driver:
            driver.get(url)

            # Find all divs with class "collapse"
            collapse_divs = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".Panel"))
            )

            # Click on each div to trigger the onclick event
            for div in collapse_divs:
                div.click()
                sleep(1)  # Wait a bit to let the content load (adjust as needed)

            # After all onclick events are triggered, scrape the loaded content
            # Note: You might want to fetch the divs again in case the DOM has changed after the clicks
            collapse_divs = driver.find_elements(By.CSS_SELECTOR, ".Panel")
            for div in collapse_divs:
                content = div.get_attribute('innerHTML')
                soup = BeautifulSoup(content, 'lxml')

                plays.append(Game.Drive(soup))

        return plays

    def prior(self):
        """
        returns the elo ratings prior to the game
        """
        return self.home.elo, self.away.elo

    def post(self):
        """
        returns the team elos after a game
        """
        winmargin = self.scoreboard.winMargin()
        homewin = 1 if winmargin[0] == self.home.name else 0 # cant handle ties


        newHome = self.home.elo.shift(self.away.elo, winmargin[1], homewin)
        newAway = self.away.elo.shift(self.home.elo, winmargin[1], 1 - homewin)

        return newHome, newAway # always add the values. The result will come back negative if necessary

    def isHome(self, team):
        """
        return true if the team is home
        """
        if not team in [self.home.name, self.away.name]:
            raise Exception(f'Team {team} not in this game: {str(self)}')
        return team == self.home.name

    def isAway(self, team):
        """
        Return true is the team is away
        """
        return not self.isHome(team)

    def eloPrediction(self, team):

        if not team in [self.home.name, self.away.name]:
            raise Exception(f'Team {team} not in game: {self.__str__()}')

        elif team == self.home.name:
            return self.home.elo.winProb(self.away.elo)

        elif team == self.away.name:
            return self.away.elo.winProb(self.home.elo)
        else:
            raise Exception('An uknown error occured in game.eloPrediction')
    
    def updateTeams(self, winMargin, homeWin):
        pass

    def teams(self):
        """
        Iterable for each team
        """
        for team in [self.home, self.away]:
            yield team

    def result(self) -> bool:
        """
        Return true if the home team won
        """
        return self.scoreboard.winner().name == self.home.name

    # MAGIC
    def __repr__(self) -> str:
        return f'{self.home}(H) V {self.away}(A)'
    def __str__(self) -> str:
        return self.__repr__()
    

class EloRating:
    def __init__(self, rating) -> None:
        self.rating = rating
        self.kFactor = 20

    def winProb(self, other) -> float:
        """
        Computes the win probability based on the teams' elo ratings
        """
        if not isinstance(other, EloRating):
            raise Exception(f'Cannot compare ELO to type {type(other)}')
        
        eloDiff = self.rating - other.rating

        return 1 / (10 ** ((-1 * eloDiff) / 400) + 1) # formulation for probability

    def shift(self, other, point_difference, game_outcome):
        """
        Update the Elo rating based on the game outcome and point difference.

        :param other: The EloRating of the opposing team.
        :param point_difference: The difference in scores between the two teams.
        :param game_outcome: 1 if this team won, 0 if they lost.
        """
        if not isinstance(other, EloRating):
            raise Exception(f'Cannot compare ELO to type {type(other)}')
        
        # Calculate the Margin of Victory Multiplier
        winner_elo_diff = self.rating - other.rating if game_outcome == 1 else other.rating - self.rating
        MoVM = math.log(abs(point_difference) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))
        
        # Calculate the Forecast Delta
        forecast_delta = game_outcome - self.winProb(other)

        # Calculate the Elo Shift
        elo_shift = self.kFactor * forecast_delta * MoVM
        

        return self.rating + elo_shift


    # MAGIC
    def __repr__(self) -> str:
        return f'{self.rating}'
    def __str__(self) -> str:
        return self.__repr__()
    
    def __add__(self, val):
        self.rating += val
        return self.rating
    

class Team:
    def __init__(self, *args, **kwargs) -> None:

        self.name = kwargs.get('name')
        self.elo = EloRating(kwargs.get('elo'))
        self.home = kwargs.get('home')

        self.restDays = 0
        self.qbRating = 0
        self.travelDistance = 0

    # MAGIC
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def __str__(self) -> str:
        return self.__repr__()

 
class Scoreboard:

    def __init__(self, home, away, homeScore, awayScore):
        self.home = home.name
        self.away = away.name
        self.homeScore = homeScore
        self.awayScore = awayScore

    def winMargin(self) -> int:
        
        if self.homeScore > self.awayScore:
            return (self.home, self.homeScore - self.awayScore)
    
        elif self.homeScore < self.awayScore:
            return (self.away, self.awayScore - self.homeScore)
    
    def winner(self):

        if self.homeScore > self.awayScore:
            return self.home
    
        elif self.homeScore < self.awayScore:
            return self.away
    

class Simluator:
    def __init__(self, year) -> None:
        self.year = year
        self.wbPath = os.path.join(SCORES_DIR, f'{year}.xlsx')
        
        if os.path.exists(self.wbPath):
            self.wb = openpyxl.load_workbook(self.wbPath)
        else:
            raise Exception('No workbook available')

    def simulate_week(self, week):
        # Load the sheet for the given week using "Wx" naming convention
        ws = self.wb[f'W{week}']

        # Get column names to indices mapping
        col_names = {cell.value: idx for idx, cell in enumerate(next(ws.iter_rows()))}  # Assuming first row contains column names

        # Store the Elo updates to apply them to the next week
        elo_updates = {}

        # For each game in the sheet:
        for row in ws.iter_rows(min_row=2):
            # Create a Game instance using data from the row
            game = Game(week=f'W{week}', year=self.year, team=row[col_names['teamname']].value)
            
            # Store the Elo changes
            elo_updates[game.home.name] = game.post()[0]
            elo_updates[game.away.name] = game.post()[1]

        # Load the next week's sheet (or create it if it doesn't exist) using "Wx" naming convention
        next_week = week + 1
        if next_week <= 18:  # Ensure we don't exceed the regular season weeks
            if f'W{next_week}' not in self.wb.sheetnames:
                ws_next = self.wb.create_sheet(title=f'W{next_week}')
            else:
                ws_next = self.wb[f'W{next_week}']

            # Update the Elo ratings for the teams in the next week's sheet
            for team, elo_change in elo_updates.items():
                for row in ws_next.iter_rows(min_row=2):
                    if row[col_names['teamname']].value == team:
                        row[col_names['elo']].value = elo_change
                        break

            # Save the workbook after making the updates
            self.wb.save(self.wbPath)


    def simulateGame(self) -> Team:
        pass

    def simulateSeason(self) -> Team:
        pass



def main():


    gameParams = {
        'week': 'W2',
        'year': '2023',
        'team': 'bills'
    }
    game = Game(**gameParams)

    print(game.isHome('bills'))
  

if __name__ == '__main__':
    from Dependencies import *
    main()

# else:
#     from src.Lib.Dependencies import *
#     print('alternate import')
