from src.Lib.Globals import *
import src.Lib.Game as Game

def GetDrives(url = None):

    if url is None:
        url = 'https://www.espn.com/nfl/playbyplay/_/gameId/401548631'

    # You can use headless mode if you don't want the browser to show up
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')

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

def generateTrainingData(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def urlt1(id):
   return f'https://www.espn.com/nfl/playbyplay/_/gameId/{id}'

def urlt2(gameNum, seasonType, year):
    return f'https://www.espn.com/nfl/scoreboard/_/week/{gameNum}/year/{year}/seasontype/{seasonType}'

def write_to_xlsx(data_dict, sheet_name, file_name='output.xlsx'):
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
    headers = ['ID', 'teamname', 'q1', 'q2', 'q3', 'q4', 'finalscore', 'home', 'win']
    ws.append(headers)

    # Iterate over the data dictionary and extract data
    for game_id, teams in data_dict.items():
        away_team, away_scores = list(teams[0].items())[0]
        home_team, home_scores = list(teams[1].items())[0]
        
        away_win = 1 if away_scores[-1] > home_scores[-1] else 0
        home_win = 1 - away_win
        
        # Write away team data
        ws.append([game_id, away_team] + list(away_scores) + [0, away_win])
        
        # Write home team data
        ws.append([game_id, home_team] + list(home_scores) + [1, home_win])

    # Save the workbook
    wb.save(file_name)
    print(f'SAVEDWB{sheet_name}')
 
def GetIDS(url = None):
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

                        li1, li2 = nested_section.ul.find_all('li', recursive=False)
                        try:
                            t1scores = []
                            for child in li1.find_all('div', recursive = False)[1]:
                                t1scores.append(int(child.text))
                            
                            t2scores = []
                            for child in li2.find_all('div', recursive = False)[1]:
                                t2scores.append(int(child.text))

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

def extract_recaps():
    base_dir = './datasets/gameScores'
    recap_dir = './datasets/gameRecaps'
    
    # Iterate over the xlsx files
    for year in range(2023, 2024):
        file_path = os.path.join(base_dir, f"{year}.xlsx")
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
                                year_dir = os.path.join(recap_dir, str(year))
                                week_dir = os.path.join(year_dir, sheet_name)
                                os.makedirs(week_dir, exist_ok=True)
                                
                                # Save the article
                                recap_file_path = os.path.join(week_dir, f"{teamname}vs{team2name}.txt")
                                with open(recap_file_path, 'w', encoding='utf-8') as file:
                                    file.write(article_text)
                    except urllib.error.HTTPError as err:
                        print(f'ERROR FROM {teamname}, {year}')
                        
                # Skip the next row as it's the second team of the same game
                i += 2

def grab():
    for i in range(1,4):
        for j in range(1,19):

            if i == 1 and j >= 4:
                continue
            if i == 3 and j >= 6:
                continue

            
            if i == 1:
                sheetName = f'P{j}'
            elif i==2:
                sheetName = f'W{j}'
            elif i == 3:
                sheetName = f'PO{j}'

            

            gameData = GetIDS(urlt2(j, i, 2023))
            write_to_xlsx(gameData, sheetName, file_name='2023.xlsx')



def main(*args):
    # parse args
    if 'pull' in args:
        extract_recaps()
        grab()
    if 'train' in args:
        pass
    if 'predict' in args:
        pass


