const BLOCK_SEARCH_TERMS = ["sex", "drugs"];

// Blacklist search terms
const SearchTermBlackList = {};
for (let i = 0; i < BLOCK_SEARCH_TERMS.length; i++) {
  SearchTermBlackList[BLOCK_SEARCH_TERMS[i]] = 0;
}

// initializing searchTermFilteringBlacklist service
const searchTermFilteringBlacklist = value => {
  const wordsArray = value.split(" ");
  let okBlacklist = false;
  for (let i = 0; i < wordsArray.length; i++) {
    if (wordsArray[i].trim() in SearchTermBlackList) okBlacklist = true;
  }
  return okBlacklist;
};

module.exports = { SearchTermBlackList, searchTermFilteringBlacklist };
