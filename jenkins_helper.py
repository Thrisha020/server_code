import os
import yaml
import requests
from git import Repo

class JenkinsCIPipeline:
    def __init__(self, jenkins_url, job_name, username, password):
        self.jenkins_url = jenkins_url
        self.job_name = job_name
        self.auth = (username, password)

    def get_crumb(self):
        """Fetch CSRF crumb for Jenkins authentication."""
        try:
            crumb_url = f"{self.jenkins_url}/crumbIssuer/api/json"
            response = requests.get(crumb_url, auth=self.auth)
            response.raise_for_status()
            return response.json().get("crumbRequestField"), response.json().get("crumb")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Jenkins crumb: {e}")
            return None, None

    def trigger_job(self, parameters):
        """Trigger a Jenkins job with parameters."""
        try:
            print("Triggering Jenkins CI pipeline...")
            crumb_field, crumb_value = self.get_crumb()
            if not crumb_field or not crumb_value:
                print("Failed to retrieve CSRF token. Exiting...")
                return False

            headers = {crumb_field: crumb_value}
            url = f"{self.jenkins_url}/job/{self.job_name}/buildWithParameters"
            response = requests.post(url, auth=self.auth, params=parameters, headers=headers)
            response.raise_for_status()
            print("Job triggered successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error triggering Jenkins job: {e}")
            return False

    def monitor_job_status(self):
        """Placeholder for monitoring Jenkins job status."""
        print("Monitoring Jenkins job status... (Implementation pending)")
        # Logic to poll Jenkins API for job completion status

class GitHandler:
    def __init__(self, repo_url):
        self.repo_url = repo_url

    def clone_repo(self, local_path):
        """Clone a repository to a local path."""
        try:
            if not os.path.exists(local_path):
                print(f"Cloning repository from {self.repo_url}...")
                Repo.clone_from(self.repo_url, local_path)
            else:
                print(f"Repository already exists at {local_path}.")
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            raise

    def checkout_branch(self, repo_path, branch_name):
        """Checkout a specific branch in the local repository."""
        try:
            repo = Repo(repo_path)
            repo.git.checkout(branch_name)
            print(f"Checked out to branch {branch_name}.")
            return True
        except Exception as e:
            print(f"Error checking out branch: {e}")
            return False

    def validate_file(self, repo_path, file_name):
        """Recursively search for a file in the repository."""
        print(f"Searching for file {file_name} in repository: {repo_path}")
        #os.walk for search in all folder
        for root, dirs, files in os.walk(repo_path):
            print(f"Searching in directory: {root}")
            if file_name in files:
                file_path = os.path.join(root, file_name)
                print(f"File {file_name} found at: {file_path}")
                return True
        
        print(f"File {file_name} does not exist in any directory of the repository.")
        return False

    def commit_and_push_changes(self, repo_path, file_name):
        """Commit and push changes to the remote repository."""
        try:
            repo = Repo(repo_path)
            repo.git.add(file_name)
            repo.index.commit(f"Automated commit for {file_name}")
            origin = repo.remote(name='origin')
            origin.push()
            print("Changes pushed to remote repository.")
            return True
        except Exception as e:
            print(f"Error pushing changes: {e}")
            return False

def load_yaml_config(yaml_file):
    """Load configuration from a YAML file."""
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"YAML file {yaml_file} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def main():
    """Main function to orchestrate the CI pipeline."""
    yaml_file = "config.yaml"
    config = load_yaml_config(yaml_file)
    if not config:
        print("Failed to load configuration. Exiting...")
        return

    repo_url = config.get("repositories", {}).get("git_url")
    if not repo_url:
        print("Repository URL not found in YAML configuration. Exiting...")
        return

    jenkins_url = "http://43.204.219.229:8080/"
    job_name = "Development/MortgageApplication-BuildFromChatbot"
    jenkins_username = "admin"
    jenkins_password = "Passw0rd"

    application_name = input("Enter Git application name: ").strip()
    branch_name = input("Enter Git branch name: ").strip()
    file_name = input("Enter source file name to build: ").strip()
    push_changes = input("Do you want to push local changes to remote? (Yes/No): ").strip().lower() == "yes"

    local_path = f"./{application_name}"
    git_handler = GitHandler(repo_url)
    git_handler.clone_repo(local_path)

    if not git_handler.checkout_branch(local_path, branch_name):
        print("Branch not found. Exiting...")
        return

    if not git_handler.validate_file(local_path, file_name):
        print("File not found in repository. Exiting...")
        return

    if push_changes:
        if not git_handler.commit_and_push_changes(local_path, file_name):
            print("Failed to push changes. Exiting...")
            return

    jenkins = JenkinsCIPipeline(jenkins_url, job_name, jenkins_username, jenkins_password)
    parameters = {'gitBranch': branch_name}
    if not jenkins.trigger_job(parameters):
        print("Failed to trigger Jenkins job. Exiting...")
        return

    jenkins.monitor_job_status()

if __name__ == "__main__":
    main()


#feature/Test-demo

#EPSCMORT.cbl