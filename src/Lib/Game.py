import bs4
import re

class Note:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], bs4.element.Tag):
            self.constructFromSoup(args[0])
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f'Sumamry: {self.description}'
    

    def constructFromSoup(self, soup):
        description = soup.find('span', class_='PlayListItem__Description')
        if description:
            self.description = description.text 
            return
        

        raise Exception('Not a note object')
        



class Play:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], bs4.element.Tag):
            self.constructFromSoup(args[0])

    ######### MAGIC ##########    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f'{self.down} & {self.distance} on {self.yardline}\nSumamry: {self.description}'

    ######### METHODS ##########
    def constructFromSoup(self, soup):
        headline = soup.find('h3', class_='PlayListItem__Headline')
        description = soup.find('span', class_='PlayListItem__Description')

        if headline and description:
            self.headline = headline.text
            self.description = description.text
            
            downDist, yardline = [elem.strip() for elem in self.headline.split('at')]
            self.down = {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}[downDist.split('&')[0].strip()]
            self.distance = int(downDist.split('&')[1].strip())

            self.yardline = yardline

            return
        
        
        raise Exception('Not a play object')


class Drive:
    def __init__(self, *args):
        self.plays = []
        self.result = ''
        self.homeScore = 0
        self.awayScore = 0
        self.homeName = ''
        self.awayName = ''
        self.possession = ''

        self.playCount = 0
        self.yards = 0
        self.time = (0,0)

        if len(args) == 1 and isinstance(args[0], bs4.element.Tag):
            self.constructFromSoup(args[0])

        

    ######### MAGIC ##########
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        possession = F'Possession: {self.possession}'
        result = f'Result: {self.result}'
        score = f'{self.homeName}: {self.homeScore}\n{self.awayName}: {self.awayScore}'
        playCount = f'Plays: {self.playCount}'
        yards = f'Yards: {self.yards}'
        time = f'Time: {self.time[0]}:{self.time[1]}'

        return '\n'.join([possession, result, score, playCount, yards, time] + list(map(str, self.plays)))

    ######### METHODS ##########
    
    def constructFromSoup(self, soup):
        result = soup.find('span', class_='AccordionHeader__Left__Drives__Headline')
        plays = soup.find('ul', class_='PlayList')
        awayName = soup.find('span', class_='AccordionHeader__Right__AwayTeam__Name')
        homeName = soup.find('span', class_='AccordionHeader__Right__HomeTeam__Name')
        homeScore = soup.find('span', class_='AccordionHeader__Right__HomeTeam__Score')
        awayScore = soup.find('span', class_='AccordionHeader__Right__AwayTeam__Score')

        description = soup.find('span', class_='AccordionHeader__Left__Drives__Description')

        possesion = soup.find('img', class_='Logo').get('alt')

        # Check if the ul_tag is found
        if plays and result:
            self.result = result.string
            self.awayName = awayName.string
            self.homeName = homeName.string
            self.awayScore = int(awayScore.string)
            self.homeScore = int(homeScore.string)
            description = description.string
            self.possession = possesion


            # Assign description information
            pattern = r"(?P<plays>\d+)\s*plays,\s*(?P<yards>\d+)\s*yards,\s*(?P<minutes>\d+):(?P<seconds>\d+)"

            match = re.search(pattern, description)

            if match:
                playCount = int(match.group('plays'))
                yards = int(match.group('yards'))
                minutes = int(match.group('minutes'))
                seconds = int(match.group('seconds'))
                
                time = (minutes, seconds)

                # assignment
                self.playCount = playCount
                self.yards = yards
                self.time = time


            # Iterate through the li elements within the ul tag
            for play in plays.find_all('li', class_='PlayListItem'):
                # Process each li element here
                # 2 types, Note and play
                try:
                    if len(list(play.children)) == 1:
                        # type is play
                        self.plays.append(Note(play))
                    elif len(list(play.children)) == 2:
                        self.plays.append(Play(play))
                except:
                    continue


