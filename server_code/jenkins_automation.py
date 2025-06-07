import os
import yaml
import requests
import jenkins
from git import Repo

class JenkinsCIPipeline:
    def __init__(self, jenkins_url, username, password):
        self.jenkins_url = jenkins_url
        self.server = jenkins.Jenkins(jenkins_url, username=username, password=password)
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

    def trigger_job(self, job_name, parameters=None):
        """Trigger a Jenkins job with parameters."""
        try:
            print(f"Triggering Jenkins job: {job_name}")
            crumb_field, crumb_value = self.get_crumb()
            if not crumb_field or not crumb_value:
                print("Failed to retrieve CSRF token. Exiting...")
                return False

            headers = {crumb_field: crumb_value}
            url = f"{self.jenkins_url}/job/{job_name}/buildWithParameters"
            response = requests.post(url, auth=self.auth, params=parameters, headers=headers)
            response.raise_for_status()
            print("Job triggered successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error triggering Jenkins job: {e}")
            return False

    def create_or_update_job(self, job_name, pipeline_script=None):
        """Create a new Jenkins job or update an existing one."""
        if self.server.job_exists(job_name):
            print(f"Job {job_name} already exists. Create new job")
            return

        config_xml = f"""
        <flow-definition plugin="workflow-job@2.40">
            <definition class="org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition" plugin="workflow-cps@2.90">
                <script>{pipeline_script or 'pipeline { agent any; stages { stage(\'Build\') { steps { echo \"Building...\" } } } }'}</script>
                <sandbox>true</sandbox>
            </definition>
        </flow-definition>
        """
        self.server.create_job(job_name, config_xml)
        print(f"Job {job_name} created successfully.")

    def monitor_job_status(self, job_name):
        """Monitor Jenkins job status."""
        print(f"Monitoring Jenkins job: {job_name}...")
        # Poll Jenkins API for job completion status (to be implemented)

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

def main():
    jenkins_url = "https://gms.genairesonance.com/"
    username = "admin"
    password = "Passw0rd"

    repo_url = input("Enter GitHub repository URL: ").strip()
    job_name = input("Enter Jenkins job name (or leave blank to create a new job): ").strip()
    branch_name = input("Enter branch name to checkout: ").strip()

    local_path = "./repo_clone"
    git_handler = GitHandler(repo_url)
    git_handler.clone_repo(local_path)
    git_handler.checkout_branch(local_path, branch_name)

    jenkins = JenkinsCIPipeline(jenkins_url, username, password)

    if not job_name:
        job_name = input("Enter new Jenkins job name: ").strip()
        custom_pipeline = input("Enter custom pipeline script (or leave blank for default): ").strip()
        jenkins.create_or_update_job(job_name, custom_pipeline or None)
    
    parameters = {'gitBranch': branch_name}
    jenkins.trigger_job(job_name, parameters)
    jenkins.monitor_job_status(job_name)

if __name__ == "__main__":
    main()
